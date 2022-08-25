// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <algorithm>
#include <cassert>
#include <unordered_map>
#include <utility>
#include <vector>

#include "rlmeta/cc/numpy_utils.h"
#include "rlmeta/cc/samplers/sampler.h"
#include "rlmeta/cc/torch_utils.h"

namespace py = pybind11;

namespace rlmeta {

class UniformSampler : public Sampler {
 public:
  UniformSampler() = default;
  explicit UniformSampler(int64_t capacity) { keys_.reserve(capacity); }

  bool Insert(int64_t key, double /*priority*/) override {
    const int64_t index = keys_.size();
    const auto [it, ret] = key_to_index_.emplace(key, index);
    if (ret) {
      keys_.push_back(key);
    }
    return ret;
  }

  py::array_t<bool> Insert(const py::array_t<int64_t>& keys,
                           double priority) override {
    py::array_t<bool> mask = utils::NumpyEmptyLike<int64_t, bool>(keys);
    InsertImpl(keys.size(), keys.data(), priority, mask.mutable_data());
    return mask;
  }

  py::array_t<bool> Insert(const py::array_t<int64_t>& keys,
                           const py::array_t<double>& /*priorities*/) override {
    assert(keys.size() == priorities.size());
    py::array_t<bool> mask = utils::NumpyEmptyLike<int64_t, bool>(keys);
    InsertImpl(keys.size(), keys.data(), nullptr, mask.mutable_data());
    return mask;
  }

  torch::Tensor Insert(const torch::Tensor& keys, double priority) override {
    assert(keys.dtype() == torch::kInt64);
    const torch::Tensor keys_contiguous = keys.contiguous();
    torch::Tensor mask = torch::empty_like(keys_contiguous, torch::kBool);
    InsertImpl(keys_contiguous.numel(), keys_contiguous.data_ptr<int64_t>(),
               priority, mask.data_ptr<bool>());
    return mask;
  }

  torch::Tensor Insert(const torch::Tensor& keys,
                       const torch::Tensor& /*priorities*/) override {
    assert(keys.dtype() == torch::kInt64);
    const torch::Tensor keys_contiguous = keys.contiguous();
    torch::Tensor mask = torch::empty_like(keys_contiguous, torch::kBool);
    InsertImpl(keys_contiguous.numel(), keys_contiguous.data_ptr<int64_t>(),
               nullptr, mask.data_ptr<bool>());
    return mask;
  }

  bool Update(int64_t key, double /*priority*/) override {
    return key_to_index_.find(key) != key_to_index_.end();
  }

  py::array_t<bool> Update(const py::array_t<int64_t>& keys,
                           double priority) override {
    py::array_t<bool> mask = utils::NumpyEmptyLike<int64_t, bool>(keys);
    UpdateImpl(keys.size(), keys.data(), priority, mask.mutable_data());
    return mask;
  }

  py::array_t<bool> Update(const py::array_t<int64_t>& keys,
                           const py::array_t<double>& /*priorities*/) override {
    assert(keys.size() == priorities.size());
    py::array_t<bool> mask = utils::NumpyEmptyLike<int64_t, bool>(keys);
    UpdateImpl(keys.size(), keys.data(), nullptr, mask.mutable_data());
    return mask;
  }

  torch::Tensor Update(const torch::Tensor& keys, double priority) override {
    assert(keys.dtype() == torch::kInt64);
    const torch::Tensor keys_contiguous = keys.contiguous();
    torch::Tensor mask = torch::empty_like(keys_contiguous, torch::kBool);
    UpdateImpl(keys_contiguous.numel(), keys_contiguous.data_ptr<int64_t>(),
               priority, mask.data_ptr<bool>());
    return mask;
  }

  torch::Tensor Update(const torch::Tensor& keys,
                       const torch::Tensor& /*priorities*/) override {
    assert(keys.dtype() == torch::kInt64);
    const torch::Tensor keys_contiguous = keys.contiguous();
    torch::Tensor mask = torch::empty_like(keys_contiguous, torch::kBool);
    UpdateImpl(keys_contiguous.numel(), keys_contiguous.data_ptr<int64_t>(),
               nullptr, mask.data_ptr<bool>());
    return mask;
  }

  bool Delete(int64_t key) override {
    const auto it = key_to_index_.find(key);
    if (it == key_to_index_.end()) {
      return false;
    }
    const size_t index = it->second;
    if (index < keys_.size() - 1) {
      std::swap(keys_[index], keys_.back());
    }
    keys_.pop_back();
    key_to_index_.erase(it);
    return true;
  }

  py::array_t<bool> Delete(const py::array_t<int64_t>& keys) {
    py::array_t<bool> mask = utils::NumpyEmptyLike<int64_t, bool>(keys);
    DeleteImpl(keys.size(), keys.data(), mask.mutable_data());
    return mask;
  }

  torch::Tensor Delete(const torch::Tensor& keys) {
    assert(keys.dtype() == torch::kInt64);
    const torch::Tensor keys_contiguous = keys.contiguous();
    torch::Tensor mask = torch::empty_like(keys_contiguous, torch::kBool);
    DeleteImpl(keys_contiguous.numel(), keys_contiguous.data_ptr<int64_t>(),
               mask.data_ptr<bool>());
    return mask;
  }

  KeysAndPriorities Sample(int64_t num) override;

  py::array_t<int64_t> DumpData() const {
    return utils::AsNumpyArray<int64_t>(keys_);
  }

  void LoadData(const py::array_t<int64_t>& arr) {
    const int64_t n = arr.size();
    keys_.assign(arr.data(), arr.data() + n);
    key_to_index_.clear();
    for (int64_t i = 0; i < n; ++i) {
      key_to_index_.emplace(keys_[i], i);
    }
  }

 protected:
  void InsertImpl(int64_t n, const int64_t* keys, double priority, bool* mask);
  void InsertImpl(int64_t n, const int64_t* keys, const double* priorities,
                  bool* mask);

  void UpdateImpl(int64_t n, const int64_t* keys, double priority, bool* mask);
  void UpdateImpl(int64_t n, const int64_t* keys, const double* priorities,
                  bool* mask);

  void DeleteImpl(int64_t n, const int64_t* keys, bool* mask);

  std::vector<int64_t> keys_;
  std::unordered_map<int64_t, int64_t> key_to_index_;
};

void DefineUniformSampler(py::module& m);

}  // namespace rlmeta
