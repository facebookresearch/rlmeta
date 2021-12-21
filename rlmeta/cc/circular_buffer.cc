// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "rlmeta/cc/circular_buffer.h"

#include <cassert>
#include <memory>

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

int64_t CircularBuffer::Append(const py::object& o) {
  const int64_t cur_size = data_.size();
  const int64_t index = cursor_;
  if (cur_size < capacity_) {
    data_.push_back(o);
  } else {
    data_[cursor_] = o;
  }
  NextCursor();
  return index;
}

py::tuple CircularBuffer::DumpData() const {
  const int64_t n = data_.size();
  py::tuple ret(n);
  for (int64_t i = 0; i < n; ++i) {
    ret[i] = data_[i];
  }
  return ret;
}

void CircularBuffer::LoadData(const py::tuple& src, int64_t cursor) {
  Reset();
  for (const auto& o : src) {
    data_.push_back(py::reinterpret_borrow<py::object>(o));
  }
  cursor_ = cursor;
}

py::tuple CircularBuffer::BatchAtImpl(int64_t n, const int64_t* index) const {
  py::tuple ret(n);
  for (int64_t i = 0; i < n; ++i) {
    ret[i] = data_.at(index[i]);
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
py::array_t<int64_t> CircularBuffer::ExtendImpl(const Sequence& src) {
  const int64_t n = src.size();
  py::array_t<int64_t> index(n);
  int64_t* index_data = index.mutable_data();
  for (int64_t i = 0; i < n; ++i) {
    index_data[i] = cursor_;
    Append(py::reinterpret_borrow<py::object>(src[i]));
  }
  return index;
}

void DefineCircularBuffer(py::module& m) {
  py::class_<CircularBuffer, std::shared_ptr<CircularBuffer>>(m,
                                                              "CircularBuffer")
      .def(py::init<int64_t>())
      .def("__len__", &CircularBuffer::Size)
      .def("size", &CircularBuffer::Size)
      .def("capacity", &CircularBuffer::capacity)
      .def("cursor", &CircularBuffer::cursor)
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
      .def("append", &CircularBuffer::Append)
      .def("extend",
           py::overload_cast<const py::tuple&>(&CircularBuffer::Extend))
      .def("extend",
           py::overload_cast<const py::list&>(&CircularBuffer::Extend))
      .def(py::pickle(
          [](const CircularBuffer& s) {
            return py::make_tuple(s.capacity(), s.DumpData(), s.cursor());
          },
          [](const py::tuple& t) {
            assert(t.size() == 3);
            const int64_t capacity = t[0].cast<int64_t>();
            const py::tuple data = py::reinterpret_borrow<py::tuple>(t[1]);
            const int64_t cursor = t[2].cast<int64_t>();
            CircularBuffer buffer(capacity);
            buffer.LoadData(data, cursor);
            return buffer;
          }));
}

}  // namespace rlmeta
