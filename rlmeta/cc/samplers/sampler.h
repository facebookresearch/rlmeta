// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <random>
#include <utility>

namespace py = pybind11;

namespace rlmeta {

using KeysAndPriorities = std::pair<py::array_t<int64_t>, py::array_t<double>>;

class Sampler {
 public:
  virtual void Seed(int64_t seed) { random_gen_.seed(seed); }

  virtual bool Insert(int64_t key, double priority) = 0;
  virtual py::array_t<bool> Insert(const py::array_t<int64_t>& keys,
                                   double priority) = 0;
  virtual py::array_t<bool> Insert(const py::array_t<int64_t>& keys,
                                   const py::array_t<double>& priorities) = 0;
  virtual torch::Tensor Insert(const torch::Tensor& keys, double priority) = 0;
  virtual torch::Tensor Insert(const torch::Tensor& keys,
                               const torch::Tensor& priorities) = 0;

  virtual bool Update(int64_t key, double priority) = 0;
  virtual py::array_t<bool> Update(const py::array_t<int64_t>& keys,
                                   double priority) = 0;
  virtual py::array_t<bool> Update(const py::array_t<int64_t>& keys,
                                   const py::array_t<double>& priorities) = 0;
  virtual torch::Tensor Update(const torch::Tensor& keys, double priority) = 0;
  virtual torch::Tensor Update(const torch::Tensor& keys,
                               const torch::Tensor& priorities) = 0;

  virtual bool Delete(int64_t key) = 0;
  virtual py::array_t<bool> Delete(const py::array_t<int64_t>& keys) = 0;
  virtual torch::Tensor Delete(const torch::Tensor& keys) = 0;

  virtual KeysAndPriorities Sample(int64_t num) = 0;

 protected:
  std::mt19937_64 random_gen_{std::random_device()()};
};

class PySampler : public Sampler {
 public:
  bool Insert(int64_t key, double priority) override {
    PYBIND11_OVERRIDE_PURE(bool, Sampler, Insert, key, priority);
  }

  py::array_t<bool> Insert(const py::array_t<int64_t>& keys,
                           double priority) override {
    PYBIND11_OVERRIDE_PURE(py::array_t<bool>, Sampler, Insert, keys, priority);
  }

  py::array_t<bool> Insert(const py::array_t<int64_t>& keys,
                           const py::array_t<double>& priorities) override {
    PYBIND11_OVERRIDE_PURE(py::array_t<bool>, Sampler, Insert, keys,
                           priorities);
  }

  torch::Tensor Insert(const torch::Tensor& keys, double priority) override {
    PYBIND11_OVERRIDE_PURE(torch::Tensor, Sampler, Insert, keys, priority);
  }

  torch::Tensor Insert(const torch::Tensor& keys,
                       const torch::Tensor& priorities) override {
    PYBIND11_OVERRIDE_PURE(torch::Tensor, Sampler, Insert, keys, priorities);
  }

  bool Update(int64_t key, double priority) override {
    PYBIND11_OVERRIDE_PURE(bool, Sampler, Update, key, priority);
  }

  py::array_t<bool> Update(const py::array_t<int64_t>& keys,
                           double priority) override {
    PYBIND11_OVERRIDE_PURE(py::array_t<bool>, Sampler, Update, keys, priority);
  }

  py::array_t<bool> Update(const py::array_t<int64_t>& keys,
                           const py::array_t<double>& priorities) override {
    PYBIND11_OVERRIDE_PURE(py::array_t<bool>, Sampler, Update, keys,
                           priorities);
  }

  torch::Tensor Update(const torch::Tensor& keys, double priority) override {
    PYBIND11_OVERRIDE_PURE(torch::Tensor, Sampler, Update, keys, priority);
  }

  torch::Tensor Update(const torch::Tensor& keys,
                       const torch::Tensor& priorities) override {
    PYBIND11_OVERRIDE_PURE(torch::Tensor, Sampler, Update, keys, priorities);
  }

  bool Delete(int64_t key) override {
    PYBIND11_OVERRIDE_PURE(bool, Sampler, Delete, key);
  }

  py::array_t<bool> Delete(const py::array_t<int64_t>& keys) override {
    PYBIND11_OVERRIDE_PURE(py::array_t<int64_t>, Sampler, Delete, keys);
  }

  torch::Tensor Delete(const torch::Tensor& keys) override {
    PYBIND11_OVERRIDE_PURE(torch::Tensor, Sampler, Delete, keys);
  }

  KeysAndPriorities Sample(int64_t num) override {
    PYBIND11_OVERRIDE_PURE(KeysAndPriorities, Sampler, Sample, num);
  }
};

void DefineSampler(py::module& m);

}  // namespace rlmeta
