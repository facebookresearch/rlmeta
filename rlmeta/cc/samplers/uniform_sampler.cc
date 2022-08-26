// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "rlmeta/cc/samplers/uniform_sampler.h"

#include <memory>
#include <random>

namespace rlmeta {

KeysAndPriorities UniformSampler::Sample(int64_t num) const {
  py::array_t<int64_t> keys(num);
  py::array_t<double> priorities(num);
  std::uniform_int_distribution<int64_t> distrib(0, keys_.size() - 1);
  int64_t* keys_data = keys.mutable_data();
  double* priorities_data = priorities.mutable_data();
  for (int64_t i = 0; i < num; ++i) {
    const int64_t index = distrib(random_gen_);
    keys_data[i] = keys_[index];
    priorities_data[i] = 1.0;
  }
  return std::make_pair<py::array_t<int64_t>, py::array_t<double>>(
      std::move(keys), std::move(priorities));
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
