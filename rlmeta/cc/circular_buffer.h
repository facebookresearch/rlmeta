// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace rlmeta {

class CircularBuffer {
 public:
  explicit CircularBuffer(int64_t capacity) : capacity_(capacity) { Reset(); }

  int64_t capacity() const { return capacity_; }

  int64_t cursor() const { return cursor_; }

  int64_t next_key() const { return next_key_; }

  int64_t Size() const { return data_.size(); }

  py::object At(int64_t key) const {
    const auto it = key_to_index_.find(key);
    return it == key_to_index_.end() ? py::none() : data_.at(it->second).second;
  }

  py::tuple At(const py::array_t<int64_t>& keys) const;

  py::tuple At(const torch::Tensor& keys) const;

  void Reset();

  std::pair<std::int64_t, int64_t> Append(const py::object& o);

  std::pair<py::array_t<int64_t>, py::array_t<int64_t>> Extend(
      const py::tuple& src) {
    return ExtendImpl(src);
  }

  std::pair<py::array_t<int64_t>, py::array_t<int64_t>> Extend(
      const py::list& src) {
    return ExtendImpl(src);
  }

  py::tuple DumpData() const;

  void LoadData(const py::tuple& src, int64_t cursor, int64_t next_key);

 protected:
  py::tuple BatchAtImpl(int64_t n, const int64_t* keys) const;

  void NextCursor();

  template <class Sequence>
  std::pair<py::array_t<int64_t>, py::array_t<int64_t>> ExtendImpl(
      const Sequence& src);

  const int64_t capacity_;
  std::vector<std::pair<int64_t, py::object>> data_;
  std::unordered_map<int64_t, int64_t> key_to_index_;
  int64_t cursor_ = 0;
  int64_t next_key_ = 0;
};

void DefineCircularBuffer(py::module& m);

}  // namespace rlmeta
