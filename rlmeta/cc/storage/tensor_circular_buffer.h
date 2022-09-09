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
#include <unordered_map>
#include <utility>
#include <vector>

#include "rlmeta/cc/storage/circular_buffer_base.h"
#include "rlmeta/cc/storage/schema.h"
#include "rlmeta/cc/utils/numpy_utils.h"

namespace py = pybind11;

namespace rlmeta {

class TensorCircularBuffer : public CircularBufferBase {
 public:
  explicit TensorCircularBuffer(int64_t capacity)
      : CircularBufferBase(capacity) {
    keys_.reserve(capacity);
  }

  TensorCircularBuffer(int64_t capacity, int64_t num_threads)
      : CircularBufferBase(capacity), num_threads_(num_threads) {
    keys_.reserve(capacity);
  }

  int64_t Size() const override { return keys_.size(); }

  int64_t num_threads() const { return num_threads_; }

  bool initialized() const { return initialized_; }

  int64_t next_key() const { return next_key_; }

  void Clear() override {
    CircularBufferBase::Clear();
    data_.clear();
    keys_.clear();
    keys_.reserve(capacity_);
    key_to_index_.clear();
    next_key_ = 0;
  }

  py::object At(int64_t key) const override;

  py::object At(const py::array_t<int64_t>& keys) const override {
    assert(index.ndim() == 1);
    return BatchedAtImpl(keys.size(), keys.data());
  }

  py::object At(const torch::Tensor& keys) const override {
    assert(keys.dtype() == torch::kInt64);
    assert(keys.dim() == 1);
    const torch::Tensor keys_contiguous = keys.contiguous();
    return BatchedAtImpl(keys_contiguous.numel(),
                         keys_contiguous.data_ptr<int64_t>());
  }

  std::pair<int64_t, std::optional<int64_t>> Append(
      const py::object& o) override;

  std::pair<py::array_t<int64_t>, py::array_t<int64_t>> Extend(
      const py::tuple& src) override {
    return ExtendImpl(src);
  }

  std::pair<py::array_t<int64_t>, py::array_t<int64_t>> Extend(
      const py::list& src) override {
    return ExtendImpl(src);
  }

  py::object DumpData() const { return RecoverNested(data_); }

  py::array_t<int64_t> DumpKeys() const { return utils::AsNumpyArray(keys_); }

  void LoadData(const py::object& data, const py::array_t<int64_t>& keys,
                int64_t cursor, int64_t next_key);

 protected:
  void Init(const py::object& o);

  py::object RecoverNested(const std::vector<torch::Tensor>& src) const;

  py::object BatchedAtImpl(int64_t n, const int64_t* keys) const;

  std::pair<int64_t, std::optional<int64_t>> Reserve();

  std::pair<std::vector<int64_t>, std::vector<int64_t>> Reserve(int64_t num);

  void InsertImpl(const std::vector<int64_t>& keys,
                  const std::vector<std::vector<torch::Tensor>>& data,
                  int64_t begin, int64_t end);

  template <class Sequence>
  std::pair<py::array_t<int64_t>, py::array_t<int64_t>> ExtendImpl(
      const Sequence& src);

  const int64_t num_threads_ = 1;

  bool initialized_ = false;
  Schema schema_;
  std::vector<torch::Tensor> data_;

  std::vector<int64_t> keys_;
  std::unordered_map<int64_t, int64_t> key_to_index_;
  int64_t next_key_ = 0;
};

void DefineTensorCircularBuffer(py::module& m);

}  // namespace rlmeta
