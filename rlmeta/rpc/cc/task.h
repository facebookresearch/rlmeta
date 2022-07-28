// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <pybind11/pybind11.h>

#include <cstdint>
#include <future>
#include <memory>
#include <string>
#include <vector>

#include "rlmeta/rpc/cc/blocking_counter.h"

namespace py = pybind11;

namespace rlmeta {
namespace rpc {

class TaskBase {
 public:
  virtual py::object Args() = 0;
  virtual py::object Kwargs() = 0;
  virtual void SetReturnValue(py::object&& return_value) = 0;
};

// https://pybind11.readthedocs.io/en/stable/advanced/classes.html#overriding-virtuals
class PyTaskBase : public TaskBase {
 public:
  using TaskBase::TaskBase;

  py::object Args() override {
    PYBIND11_OVERRIDE_PURE(py::object, TaskBase, Args);
  }

  py::object Kwargs() override {
    PYBIND11_OVERRIDE_PURE(py::object, TaskBase, Kwargs);
  }

  void SetReturnValue(py::object&& return_value) override {
    PYBIND11_OVERRIDE_PURE(void, TaskBase, SetReturnValue, return_value);
  }
};

class Task : public TaskBase {
 public:
  Task(const std::string& args, const std::string& kwargs)
      : args_(args), kwargs_(kwargs) {}

  py::object Args() override { return py::bytes(std::move(args_)); }
  py::object Kwargs() override { return py::bytes(std::move(kwargs_)); }

  std::future<std::string> Future() { return promise_.get_future(); }

  void SetReturnValue(py::object&& return_value) override {
    promise_.set_value(
        py::reinterpret_borrow<py::bytes>(std::move(return_value)));
  }

 protected:
  std::string args_;
  std::string kwargs_;
  std::promise<std::string> promise_;
};

class BatchedTask : public TaskBase {
 public:
  explicit BatchedTask(int64_t capacity)
      : capacity_(capacity), num_to_wait_(capacity) {
    batch_.reserve(capacity_);
  }

  int64_t capacity() const { return capacity_; }
  int64_t BatchSize() const { return batch_.size(); }

  bool Empty() const { return batch_.empty(); }
  bool Full() const { return static_cast<int64_t>(batch_.size()) == capacity_; }

  py::object Args() override;
  py::object Kwargs() override;

  void SetReturnValue(py::object&& return_value) override;

  std::future<std::string> Add(const std::string& args,
                               const std::string& kwargs);

  void Wait() { num_to_wait_.Wait(); }

 protected:
  const int64_t capacity_;
  std::vector<Task> batch_;

  BlockingCounter num_to_wait_;
};

void DefineTaskBase(py::module& m);
void DefineTask(py::module& m);
void DefineBatchedTask(py::module& m);

}  // namespace rpc
}  // namespace rlmeta
