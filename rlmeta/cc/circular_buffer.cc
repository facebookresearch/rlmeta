// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "rlmeta/cc/circular_buffer.h"

#include <pybind11/stl.h>

#include <cassert>
#include <memory>

#include "rlmeta/cc/numpy_utils.h"

namespace rlmeta {

py::tuple CircularBuffer::At(const py::array_t<int64_t>& index) const {
  assert(index.ndim() == 1);
  return BatchAtImpl(index.size(), index.data());
}

py::tuple CircularBuffer::At(const torch::Tensor& index) const {
  assert(index.dtype() == torch::kInt64);
  assert(index.dim() == 1);
  const torch::Tensor index_contiguous = index.contiguous();
  return BatchAtImpl(index_contiguous.numel(),
                     index_contiguous.data_ptr<int64_t>());
}

void CircularBuffer::Reset() {
  data_.clear();
  data_.reserve(capacity_);
  cursor_ = 0;
}

std::pair<int64_t, int64_t> CircularBuffer::Append(const py::object& o) {
  const int64_t cur_size = data_.size();
  const int64_t new_key = next_key_;
  int64_t old_key = -1;
  key_to_index_[next_key_] = cursor_;
  if (cur_size < capacity_) {
    data_.emplace_back(next_key_, o);
  } else {
    old_key = data_[cursor_].first;
    data_[cursor_] = std::make_pair(next_key_, o);
  }
  NextCursor();
  ++next_key_;
  return std::make_pair(new_key, old_key);
}

py::tuple CircularBuffer::DumpData() const {
  const int64_t n = data_.size();
  py::tuple ret(n);
  for (int64_t i = 0; i < n; ++i) {
    ret[i] = py::make_tuple(data_[i].first, data_[i].second);
  }
  return ret;
}

void CircularBuffer::LoadData(const py::tuple& src, int64_t cursor,
                              int64_t next_key) {
  Reset();
  for (const auto o : src) {
    const auto cur = py::reinterpret_borrow<py::tuple>(o);
    data_.emplace_back(cur[0].cast<int64_t>(),
                       py::reinterpret_borrow<py::object>(cur[1]));
  }
  cursor_ = cursor;
  next_key_ = next_key;
}

py::tuple CircularBuffer::BatchAtImpl(int64_t n, const int64_t* keys) const {
  py::tuple ret(n);
  for (int64_t i = 0; i < n; ++i) {
    ret[i] = At(keys[i]);
  }
  return ret;
}

void CircularBuffer::NextCursor() {
  ++cursor_;
  if (cursor_ == capacity_) {
    cursor_ = 0;
  }
}

template <class Sequence>
std::pair<py::array_t<int64_t>, py::array_t<int64_t>>
CircularBuffer::ExtendImpl(const Sequence& src) {
  const int64_t n = src.size();
  std::vector<int64_t> new_keys;
  std::vector<int64_t> old_keys;
  new_keys.reserve(n);
  old_keys.reserve(n);
  for (int64_t i = 0; i < n; ++i) {
    const auto [new_key, old_key] =
        Append(py::reinterpret_borrow<py::object>(src[i]));
    new_keys.push_back(new_key);
    if (old_key >= 0) {
      old_keys.push_back(old_key);
    }
  }
  return std::make_pair<py::array_t<int64_t>, py::array_t<int64_t>>(
      utils::AsNumpyArray<int64_t>(std::move(new_keys)),
      utils::AsNumpyArray<int64_t>(std::move(old_keys)));
}

void DefineCircularBuffer(py::module& m) {
  py::class_<CircularBuffer, std::shared_ptr<CircularBuffer>>(m,
                                                              "CircularBuffer")
      .def(py::init<int64_t>())
      .def_property_readonly("size", &CircularBuffer::Size)
      .def_property_readonly("capacity", &CircularBuffer::capacity)
      .def_property_readonly("cursor", &CircularBuffer::cursor)
      .def_property_readonly("next_key", &CircularBuffer::next_key)
      .def("__len__", &CircularBuffer::Size)
      .def("__getitem__",
           py::overload_cast<int64_t>(&CircularBuffer::At, py::const_))
      .def("__getitem__", py::overload_cast<const py::array_t<int64_t>&>(
                              &CircularBuffer::At, py::const_))
      .def("__getitem__", py::overload_cast<const torch::Tensor&>(
                              &CircularBuffer::At, py::const_))
      .def("at", py::overload_cast<int64_t>(&CircularBuffer::At, py::const_))
      .def("at", py::overload_cast<const py::array_t<int64_t>&>(
                     &CircularBuffer::At, py::const_))
      .def("at", py::overload_cast<const torch::Tensor&>(&CircularBuffer::At,
                                                         py::const_))
      .def("reset", &CircularBuffer::Reset)
      .def("append",
           [](CircularBuffer& buffer, const py::object& o) {
             const auto [new_key, old_key] = buffer.Append(o);
             return old_key < 0 ? py::make_tuple(new_key, py::none())
                                : py::make_tuple(new_key, old_key);
           })
      .def("extend",
           py::overload_cast<const py::tuple&>(&CircularBuffer::Extend))
      .def("extend",
           py::overload_cast<const py::list&>(&CircularBuffer::Extend))
      .def(py::pickle(
          [](const CircularBuffer& s) {
            return py::make_tuple(s.capacity(), s.DumpData(), s.cursor(),
                                  s.next_key());
          },
          [](const py::tuple& t) {
            assert(t.size() == 4);
            const int64_t capacity = t[0].cast<int64_t>();
            const py::tuple data = py::reinterpret_borrow<py::tuple>(t[1]);
            const int64_t cursor = t[2].cast<int64_t>();
            const int64_t next_key = t[3].cast<int64_t>();
            CircularBuffer buffer(capacity);
            buffer.LoadData(data, cursor, next_key);
            return buffer;
          }));
}

}  // namespace rlmeta
