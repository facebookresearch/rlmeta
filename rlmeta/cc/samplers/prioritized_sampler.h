// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <cassert>
#include <cmath>
#include <unordered_map>
#include <utility>
#include <vector>

#include "rlmeta/cc/samplers/sampler.h"
#include "rlmeta/cc/utils/numpy_utils.h"
#include "rlmeta/cc/utils/segment_tree.h"
#include "rlmeta/cc/utils/torch_utils.h"

namespace py = pybind11;

namespace rlmeta {

// The implementation of UniformSampler is adapted from UniformSelector in
// DeepMind's Reverb project.
//
// https://github.com/deepmind/reverb/blob/1a9babc151a2ff67b8c6d190dd92f041f42a560a/reverb/cc/selectors/prioritized.h
//
// Copyright 2019 DeepMind Technologies Limited.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

class PrioritizedSampler : public Sampler {
 public:
  static constexpr int64_t kDefaultCapacity = 65536;
  static constexpr double kDefaultPriorityExponent = 1.0;
  static constexpr double kDefaultEps = 1e-8;

  PrioritizedSampler() : sum_tree_(capacity_) { keys_.reserve(capacity_); }

  PrioritizedSampler(int64_t capacity, double priority_exponent, double eps)
      : capacity_(capacity),
        priority_exponent_(priority_exponent),
        eps_(eps),
        sum_tree_(capacity_) {
    keys_.reserve(capacity_);
  }

  int64_t capacity() const { return capacity_; }

  double priority_exponent() const { return priority_exponent_; }

  double eps() const { return eps_; }

  int64_t Size() const override { return keys_.size(); }

  void Reset() override {
    keys_.clear();
    key_to_index_.clear();
    sum_tree_.Reset();
  }

  void Reset(int64_t seed) override {
    random_gen_.seed(seed);
    keys_.clear();
    key_to_index_.clear();
    sum_tree_.Reset();
  }

  bool Insert(int64_t key, double priority) override {
    const int64_t index = keys_.size();
    const auto [it, ret] = key_to_index_.emplace(key, index);
    if (!ret) {
      return false;
    }
    if (index >= capacity_) {
      capacity_ <<= 1;
      sum_tree_.Resize(capacity_);
    }
    keys_.push_back(key);
    sum_tree_.Update(index, ComputePriority(priority));
    return true;
  }

  py::array_t<bool> Insert(const py::array_t<int64_t>& keys,
                           double priority) override {
    py::array_t<bool> mask = utils::NumpyEmptyLike<int64_t, bool>(keys);
    InsertImpl(keys.size(), keys.data(), priority, mask.mutable_data());
    return mask;
  }

  py::array_t<bool> Insert(const py::array_t<int64_t>& keys,
                           const py::array_t<double>& priorities) override {
    assert(keys.size() == priorities.size());
    py::array_t<bool> mask = utils::NumpyEmptyLike<int64_t, bool>(keys);
    InsertImpl(keys.size(), keys.data(), priorities.data(),
               mask.mutable_data());
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
                       const torch::Tensor& priorities) override {
    assert(keys.dtype() == torch::kInt64);
    assert(priorities.dtype() == torch::kDouble);
    const torch::Tensor keys_contiguous = keys.contiguous();
    const torch::Tensor priorities_contiguous = priorities.contiguous();
    torch::Tensor mask = torch::empty_like(keys_contiguous, torch::kBool);
    InsertImpl(keys_contiguous.numel(), keys_contiguous.data_ptr<int64_t>(),
               priorities_contiguous.data_ptr<double>(), mask.data_ptr<bool>());
    return mask;
  }

  bool Update(int64_t key, double priority) override {
    const auto it = key_to_index_.find(key);
    if (it == key_to_index_.end()) {
      return false;
    }
    sum_tree_.Update(it->second, ComputePriority(priority));
    return true;
  }

  py::array_t<bool> Update(const py::array_t<int64_t>& keys,
                           double priority) override {
    py::array_t<bool> mask = utils::NumpyEmptyLike<int64_t, bool>(keys);
    UpdateImpl(keys.size(), keys.data(), priority, mask.mutable_data());
    return mask;
  }

  py::array_t<bool> Update(const py::array_t<int64_t>& keys,
                           const py::array_t<double>& priorities) override {
    assert(keys.size() == priorities.size());
    py::array_t<bool> mask = utils::NumpyEmptyLike<int64_t, bool>(keys);
    UpdateImpl(keys.size(), keys.data(), priorities.data(),
               mask.mutable_data());
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
                       const torch::Tensor& priorities) override {
    assert(keys.dtype() == torch::kInt64);
    assert(priorities.dtype() == torch::kDouble);
    const torch::Tensor keys_contiguous = keys.contiguous();
    const torch::Tensor priorities_contiguous = priorities.contiguous();
    torch::Tensor mask = torch::empty_like(keys_contiguous, torch::kBool);
    UpdateImpl(keys_contiguous.numel(), keys_contiguous.data_ptr<int64_t>(),
               priorities_contiguous.data_ptr<double>(), mask.data_ptr<bool>());
    return mask;
  }

  bool Delete(int64_t key) override {
    const auto it = key_to_index_.find(key);
    if (it == key_to_index_.end()) {
      return false;
    }
    const int64_t index = it->second;
    const int64_t last_index = keys_.size() - 1;
    if (index < last_index) {
      const int64_t last_key = keys_.back();
      keys_[index] = last_key;
      key_to_index_[last_key] = index;
      sum_tree_.Update(index, sum_tree_.At(last_index));
    }
    keys_.pop_back();
    key_to_index_.erase(it);
    sum_tree_.Update(last_index, sum_tree_.identity_element());
    return true;
  }

  py::array_t<bool> Delete(const py::array_t<int64_t>& keys) override {
    py::array_t<bool> mask = utils::NumpyEmptyLike<int64_t, bool>(keys);
    DeleteImpl(keys.size(), keys.data(), mask.mutable_data());
    return mask;
  }

  torch::Tensor Delete(const torch::Tensor& keys) override {
    assert(keys.dtype() == torch::kInt64);
    const torch::Tensor keys_contiguous = keys.contiguous();
    torch::Tensor mask = torch::empty_like(keys_contiguous, torch::kBool);
    DeleteImpl(keys_contiguous.numel(), keys_contiguous.data_ptr<int64_t>(),
               mask.data_ptr<bool>());
    return mask;
  }

  py::array_t<int64_t> DumpKeys() const {
    return utils::AsNumpyArray<int64_t>(keys_);
  }

  void LoadKeys(const py::array_t<int64_t>& arr) {
    const int64_t n = arr.size();
    keys_.assign(arr.data(), arr.data() + n);
    key_to_index_.clear();
    for (int64_t i = 0; i < n; ++i) {
      key_to_index_.emplace(keys_[i], i);
    }
  }

  py::array_t<double> DumpPriorities() const;

  void LoadPriorities(const py::array_t<double>& priorities) {
    sum_tree_.Assign(priorities.data(), priorities.data() + priorities.size());
  }

 protected:
  KeysAndProbabilities SampleWithReplacement(int64_t num_samples) override;

  KeysAndProbabilities SampleWithoutReplacement(int64_t num_samples) override;

  double ComputePriority(double priority) const {
    return std::pow(priority + eps_, priority_exponent_);
  }

  void InsertImpl(int64_t n, const int64_t* keys, double priority, bool* mask);
  void InsertImpl(int64_t n, const int64_t* keys, const double* priorities,
                  bool* mask);

  void UpdateImpl(int64_t n, const int64_t* keys, double priority, bool* mask);
  void UpdateImpl(int64_t n, const int64_t* keys, const double* priorities,
                  bool* mask);

  void DeleteImpl(int64_t n, const int64_t* keys, bool* mask);

  int64_t capacity_ = kDefaultCapacity;
  const double priority_exponent_ = kDefaultPriorityExponent;
  const double eps_ = kDefaultEps;
  std::vector<int64_t> keys_;
  std::unordered_map<int64_t, int64_t> key_to_index_;
  SumSegmentTree<double> sum_tree_;
};

void DefinePrioritizedSampler(py::module& m);

}  // namespace rlmeta
