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
#include "rlmeta/rpc/cc/rpc_utils.h"
#include "rpc.pb.h"

namespace py = pybind11;

namespace rlmeta {
namespace rpc {

class Task {
 public:
  Task() = default;

  Task(const NestedData& args, const NestedData& kwargs)
      : args_(args), kwargs_(kwargs) {}

  Task(NestedData&& args, NestedData&& kwargs)
      : args_(std::move(args)), kwargs_(std::move(kwargs)) {}

  virtual py::object Args() {
    return rpc_utils::NestedDataToPython(std::move(args_));
  }

  virtual py::object Kwargs() {
    return rpc_utils::NestedDataToPython(std::move(kwargs_));
  }

  std::future<NestedData> Future() { return promise_.get_future(); }

  virtual void SetReturnValue(const py::object& return_value) {
    promise_.set_value(rpc_utils::PythonToNestedData(return_value));
  }

 protected:
  NestedData args_;
  NestedData kwargs_;
  std::promise<NestedData> promise_;
};

class BatchedTask : public Task {
 public:
  explicit BatchedTask(int64_t capacity)
      : capacity_(capacity), num_to_wait_(capacity) {
    promises_.reserve(capacity);
  }

  int64_t capacity() const { return capacity_; }
  int64_t batch_size() const { return batch_size_; }

  bool Empty() const { return batch_size_ == 0; }
  bool Full() const { return batch_size_ == capacity_; }

  void SetReturnValue(const py::object& return_value) override;

  std::future<NestedData> Add(const NestedData& args, const NestedData& kwargs);
  std::future<NestedData> Add(NestedData&& args, NestedData&& kwargs);

  void Wait() { num_to_wait_.Wait(); }

 protected:
  const int64_t capacity_;
  int64_t batch_size_ = 0;
  std::vector<std::promise<NestedData>> promises_;

  BlockingCounter num_to_wait_;
};

void DefineTaskBase(py::module& m);
void DefineTask(py::module& m);
void DefineBatchedTask(py::module& m);

}  // namespace rpc
}  // namespace rlmeta
