// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <vector>

namespace py = pybind11;

namespace rlmeta {

class CircularBuffer {
 public:
  CircularBuffer(int64_t capacity) : capacity_(capacity) { Reset(); }

  int64_t capacity() const { return capacity_; }

  int64_t cursor() const { return cursor_; }

  int64_t Size() const { return data_.size(); }

  const py::object& At(int64_t index) const { return data_.at(index); }

  py::tuple At(const py::array_t<int64_t>& index) const;

  py::tuple At(const torch::Tensor& index) const;

  void Reset();

  int64_t Append(const py::object& o);

  py::array_t<int64_t> Extend(const py::tuple& src) { return ExtendImpl(src); }

  py::array_t<int64_t> Extend(const py::list& src) { return ExtendImpl(src); }

  py::tuple DumpData() const;

  void LoadData(const py::tuple& src, int64_t cursor);

 protected:
  py::tuple BatchAtImpl(int64_t n, const int64_t* index) const;

  void NextCursor();

  template <class Sequence>
  py::array_t<int64_t> ExtendImpl(const Sequence& src);

  const int64_t capacity_;
  std::vector<py::object> data_;
  int64_t cursor_ = 0;
};

void DefineCircularBuffer(py::module& m);

}  // namespace rlmeta
