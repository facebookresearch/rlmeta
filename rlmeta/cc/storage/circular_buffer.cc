// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "rlmeta/cc/storage/circular_buffer.h"

#include <pybind11/stl.h>

#include <cassert>
#include <memory>

#include "rlmeta/cc/utils/numpy_utils.h"

namespace rlmeta {

std::pair<int64_t, std::optional<int64_t>> CircularBuffer::Append(
    const py::object& o) {
  const int64_t cur_size = data_.size();
  const int64_t new_key = next_key_;
  int64_t old_key = -1;
  key_to_index_.emplace(new_key, cursor_);
  if (cur_size < capacity_) {
    data_.emplace_back(new_key, o);
  } else {
    old_key = data_[cursor_].first;
    data_[cursor_] = std::make_pair(new_key, o);
    key_to_index_.erase(old_key);
  }
  NextCursor();
  ++next_key_;
  return std::make_pair(
      new_key, old_key == -1 ? std::nullopt : std::make_optional(old_key));
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
  Clear();
  for (const auto o : src) {
    const auto cur = py::reinterpret_borrow<py::tuple>(o);
    const int64_t key = cur[0].cast<int64_t>();
    const int64_t index = data_.size();
    data_.emplace_back(key, py::reinterpret_borrow<py::object>(cur[1]));
    key_to_index_.emplace(key, index);
  }
  cursor_ = cursor;
  next_key_ = next_key;
}

py::tuple CircularBuffer::BatchedAtImpl(int64_t n, const int64_t* indices,
                                        int64_t* keys) const {
  py::tuple values(n);
  for (int64_t i = 0; i < n; ++i) {
    const int64_t index = AbsoluteIndex(indices[i]);
    const auto& cur = data_.at(index);
    keys[i] = cur.first;
    values[i] = cur.second;
  }
  return values;
}

py::tuple CircularBuffer::BatchedGetImpl(int64_t n, const int64_t* keys) const {
  py::tuple ret(n);
  for (int64_t i = 0; i < n; ++i) {
    ret[i] = Get(keys[i]);
  }
  return ret;
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
    if (old_key.has_value()) {
      old_keys.push_back(*old_key);
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
      .def("reset", &CircularBuffer::Reset)
      .def("clear", &CircularBuffer::Clear)
      .def("front", &CircularBuffer::Front)
      .def("back", &CircularBuffer::Back)
      .def("at", py::overload_cast<int64_t>(&CircularBuffer::At, py::const_))
      .def("at", py::overload_cast<const py::array_t<int64_t>&>(
                     &CircularBuffer::At, py::const_))
      .def("at", py::overload_cast<const torch::Tensor&>(&CircularBuffer::At,
                                                         py::const_))
      .def("get", py::overload_cast<int64_t>(&CircularBuffer::Get, py::const_))
      .def("get", py::overload_cast<const py::array_t<int64_t>&>(
                      &CircularBuffer::Get, py::const_))
      .def("get", py::overload_cast<const torch::Tensor&>(&CircularBuffer::Get,
                                                          py::const_))
      .def("append", &CircularBuffer::Append)
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
