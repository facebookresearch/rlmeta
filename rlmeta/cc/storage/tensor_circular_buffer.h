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

  void Reset() override {
    Clear();
    schema_.Reset();
    data_.clear();
    initialized_ = false;
  }

  void Clear() override {
    CircularBufferBase::Clear();
    keys_.clear();
    keys_.reserve(capacity_);
    key_to_index_.clear();
    next_key_ = 0;
  }

  std::pair<int64_t, py::object> At(int64_t index) const override {
    index = AbsoluteIndex(index);
    return {keys_.at(index), AtImpl(index)};
  }

  std::pair<py::array_t<int64_t>, py::object> At(
      const py::array_t<int64_t>& indices) const override {
    assert(indices.ndim() == 1);
    const int64_t n = indices.size();
    const std::vector<int64_t> abs_indices = AbsoluteIndices(n, indices.data());
    py::array_t<int64_t> keys(n);
    BatchedKeyAtImpl(n, abs_indices.data(), keys.mutable_data());
    py::object values = BatchedValueAtImpl(n, abs_indices.data());
    return std::make_pair<py::array_t<int64_t>, py::object>(std::move(keys),
                                                            std::move(values));
  }

  std::pair<torch::Tensor, py::object> At(
      const torch::Tensor& indices) const override {
    assert(indices.dtype() == torch::kInt64);
    assert(indices.dim() == 1);
    const int64_t n = indices.numel();
    const torch::Tensor indices_contiguous = indices.contiguous();
    const std::vector<int64_t> abs_indices =
        AbsoluteIndices(n, indices_contiguous.data_ptr<int64_t>());
    torch::Tensor keys = torch::empty_like(indices_contiguous);
    BatchedKeyAtImpl(n, abs_indices.data(), keys.data_ptr<int64_t>());
    py::object values = BatchedValueAtImpl(n, abs_indices.data());
    return std::make_pair<torch::Tensor, py::object>(std::move(keys),
                                                     std::move(values));
  }

  py::object Get(int64_t key) const override {
    const auto it = key_to_index_.find(key);
    return it == key_to_index_.end() ? py::none() : AtImpl(it->second);
  }

  py::object Get(const py::array_t<int64_t>& keys) const override {
    assert(index.ndim() == 1);
    return BatchedGetImpl(keys.size(), keys.data());
  }

  py::object Get(const torch::Tensor& keys) const override {
    assert(keys.dtype() == torch::kInt64);
    assert(keys.dim() == 1);
    const torch::Tensor keys_contiguous = keys.contiguous();
    return BatchedGetImpl(keys_contiguous.numel(),
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

  std::vector<int64_t> AbsoluteIndices(int64_t n, const int64_t* indices) const;

  py::object AtImpl(int64_t index) const;

  void BatchedKeyAtImpl(int64_t n, const int64_t* indices, int64_t* keys) const;

  py::object BatchedValueAtImpl(int64_t n, const int64_t* indices) const;

  py::object BatchedGetImpl(int64_t n, const int64_t* keys) const;

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
