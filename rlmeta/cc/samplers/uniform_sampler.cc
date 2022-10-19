// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "rlmeta/cc/samplers/uniform_sampler.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <random>

namespace rlmeta {

KeysAndProbabilities UniformSampler::SampleWithReplacement(
    int64_t num_samples) {
  const int64_t n = keys_.size();
  std::uniform_int_distribution<int64_t> distrib(0, n - 1);
  py::array_t<int64_t> keys(num_samples);
  py::array_t<double> probabilities(num_samples);
  int64_t* keys_data = keys.mutable_data();
  double* probabilities_data = probabilities.mutable_data();
  std::fill(probabilities_data, probabilities_data + num_samples,
            1.0 / static_cast<double>(n));
  for (int64_t i = 0; i < num_samples; ++i) {
    const int64_t index = distrib(random_gen_);
    keys_data[i] = keys_[index];
  }
  return std::make_pair<py::array_t<int64_t>, py::array_t<double>>(
      std::move(keys), std::move(probabilities));
}

KeysAndProbabilities UniformSampler::SampleWithoutReplacement(
    int64_t num_samples) {
  std::uniform_int_distribution<int64_t> distrib;
  using ParamType = std::uniform_int_distribution<int64_t>::param_type;

  const int64_t n = keys_.size();
  if (num_samples > n) {
    std::cerr << "[UniformSampler] Cannot take a larger sample than population "
                 "when \'replacement=False\'"
              << std::endl;
    assert(false);
  }
  std::vector<int64_t> sampled_indices;
  sampled_indices.reserve(num_samples);

  py::array_t<int64_t> keys(num_samples);
  py::array_t<double> probabilities(num_samples);
  int64_t* keys_data = keys.mutable_data();
  double* probabilities_data = probabilities.mutable_data();
  std::fill(probabilities_data, probabilities_data + num_samples,
            1.0 / static_cast<double>(n));
  for (int64_t i = 0; i < num_samples; ++i) {
    const int64_t index = distrib(random_gen_, ParamType(0, n - i - 1));
    sampled_indices.push_back(index);
    keys_data[i] = keys_[index];
    std::swap(keys_[n - i - 1], keys_[index]);
  }
  // Recover keys_.
  for (int64_t i = num_samples - 1; i >= 0; --i) {
    std::swap(keys_[n - i - 1], keys_[sampled_indices[i]]);
  }

  return std::make_pair<py::array_t<int64_t>, py::array_t<double>>(
      std::move(keys), std::move(probabilities));
}

void UniformSampler::LoadKeys(const py::array_t<int64_t>& arr) {
  const int64_t n = arr.size();
  keys_.assign(arr.data(), arr.data() + n);
  key_to_index_.clear();
  for (int64_t i = 0; i < n; ++i) {
    key_to_index_.emplace(keys_[i], i);
  }
}

void UniformSampler::InsertImpl(int64_t n, const int64_t* keys, double priority,
                                bool* mask) {
  for (int64_t i = 0; i < n; ++i) {
    mask[i] = Insert(keys[i], priority);
  }
}

void UniformSampler::InsertImpl(int64_t n, const int64_t* keys,
                                const double* /*priorities*/, bool* mask) {
  for (int64_t i = 0; i < n; ++i) {
    mask[i] = Insert(keys[i], /*priority=*/1.0);
  }
}

void UniformSampler::UpdateImpl(int64_t n, const int64_t* keys, double priority,
                                bool* mask) {
  for (int64_t i = 0; i < n; ++i) {
    mask[i] = Update(keys[i], priority);
  }
}

void UniformSampler::UpdateImpl(int64_t n, const int64_t* keys,
                                const double* /*priorities*/, bool* mask) {
  for (int64_t i = 0; i < n; ++i) {
    mask[i] = Update(keys[i], /*priority=*/1.0);
  }
}

void UniformSampler::DeleteImpl(int64_t n, const int64_t* keys, bool* mask) {
  for (int64_t i = 0; i < n; ++i) {
    mask[i] = Delete(keys[i]);
  }
}

void DefineUniformSampler(py::module& m) {
  py::class_<UniformSampler, Sampler, std::shared_ptr<UniformSampler>>(
      m, "UniformSampler")
      .def(py::init<>())
      .def(py::pickle(
          [](const UniformSampler& sampler) { return sampler.DumpKeys(); },
          [](const py::array_t<int64_t>& arr) {
            UniformSampler sampler;
            sampler.LoadKeys(arr);
            return sampler;
          }));
}

}  // namespace rlmeta
