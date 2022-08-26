// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "rlmeta/cc/samplers/prioritized_sampler.h"

#include <memory>
#include <random>

namespace rlmeta {

KeysAndPriorities PrioritizedSampler::Sample(int64_t num) const {
  py::array_t<int64_t> keys(num);
  py::array_t<double> priorities(num);
  const double sum = sum_tree_.Query(0, sum_tree_.size());
  std::uniform_real_distribution<double> distrib(0, sum);
  int64_t* keys_data = keys.mutable_data();
  double* priorities_data = priorities.mutable_data();
  for (int64_t i = 0; i < num; ++i) {
    const double mass = distrib(random_gen_);
    const int64_t index = sum_tree_.ScanLowerBound(mass);
    keys_data[i] = keys_[index];
    priorities_data[i] = sum_tree_.At(index);
  }
  return std::make_pair<py::array_t<int64_t>, py::array_t<double>>(
      std::move(keys), std::move(priorities));
}

py::array_t<double> PrioritizedSampler::DumpPriorities() const {
  const int64_t n = sum_tree_.size();
  py::array_t<double> priorities(n);
  double* priorities_data = priorities.mutable_data();
  for (int64_t i = 0; i < n; ++i) {
    priorities_data[i] = sum_tree_.At(i);
  }
  return priorities;
}

void PrioritizedSampler::InsertImpl(int64_t n, const int64_t* keys,
                                    double priority, bool* mask) {
  for (int64_t i = 0; i < n; ++i) {
    mask[i] = Insert(keys[i], priority);
  }
}

void PrioritizedSampler::InsertImpl(int64_t n, const int64_t* keys,
                                    const double* priorities, bool* mask) {
  for (int64_t i = 0; i < n; ++i) {
    mask[i] = Insert(keys[i], priorities[i]);
  }
}

void PrioritizedSampler::UpdateImpl(int64_t n, const int64_t* keys,
                                    double priority, bool* mask) {
  for (int64_t i = 0; i < n; ++i) {
    mask[i] = Update(keys[i], priority);
  }
}

void PrioritizedSampler::UpdateImpl(int64_t n, const int64_t* keys,
                                    const double* priorities, bool* mask) {
  for (int64_t i = 0; i < n; ++i) {
    mask[i] = Update(keys[i], priorities[i]);
  }
}

void PrioritizedSampler::DeleteImpl(int64_t n, const int64_t* keys,
                                    bool* mask) {
  for (int64_t i = 0; i < n; ++i) {
    mask[i] = Delete(keys[i]);
  }
}

void DefinePrioritizedSampler(py::module& m) {
  py::class_<PrioritizedSampler, Sampler, std::shared_ptr<PrioritizedSampler>>(
      m, "PrioritizedSampler")
      .def(py::init<int64_t, double, double>(),
           py::arg("capacity") = PrioritizedSampler::kDefaultCapacity,
           py::arg("priority_exponent") =
               PrioritizedSampler::kDefaultPriorityExponent,
           py::arg("eps") = PrioritizedSampler::kDefaultEps)
      .def_property_readonly("capacity", &PrioritizedSampler::capacity)
      .def_property_readonly("priority_exponent",
                             &PrioritizedSampler::priority_exponent)
      .def_property_readonly("eps", &PrioritizedSampler::eps)
      .def(py::pickle(
          [](const PrioritizedSampler& sampler) {
            return py::make_tuple(sampler.capacity(),
                                  sampler.priority_exponent(), sampler.eps(),
                                  sampler.DumpKeys(), sampler.DumpPriorities());
          },
          [](const py::tuple& src) {
            assert(src.size() == 5);
            const int64_t capacity = src[0].cast<int64_t>();
            const double priority_exponent = src[1].cast<double>();
            const double eps = src[2].cast<double>();
            PrioritizedSampler sampler(capacity, priority_exponent, eps);
            sampler.LoadKeys(src[3].cast<py::array_t<int64_t>>());
            sampler.LoadPriorities(src[4].cast<py::array_t<double>>());
            return sampler;
          }));
}

}  // namespace rlmeta
