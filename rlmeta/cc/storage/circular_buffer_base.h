// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <cstdint>
#include <optional>

namespace rlmeta {

class CircularBufferBase {
 public:
  explicit CircularBufferBase(int64_t capacity) : capacity_(capacity) {}

  int64_t capacity() const { return capacity_; }

  int64_t cursor() const { return cursor_; }

  virtual int64_t Size() const = 0;

  virtual void Reset() { Clear(); }

  virtual void Clear() { cursor_ = 0; }

  virtual py::object At(int64_t key) const = 0;
  virtual py::object At(const py::array_t<int64_t>& keys) const = 0;
  virtual py::object At(const torch::Tensor& keys) const = 0;

  virtual std::pair<int64_t, std::optional<int64_t>> Append(
      const py::object& o) = 0;

  virtual std::pair<py::array_t<int64_t>, py::array_t<int64_t>> Extend(
      const py::tuple& src) = 0;
  virtual std::pair<py::array_t<int64_t>, py::array_t<int64_t>> Extend(
      const py::list& src) = 0;

 protected:
  void NextCursor() {
    ++cursor_;
    if (cursor_ == capacity_) {
      cursor_ = 0;
    }
  }

  const int64_t capacity_;
  int64_t cursor_ = 0;
};

}  // namespace rlmeta
