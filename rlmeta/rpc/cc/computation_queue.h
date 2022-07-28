// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <pybind11/pybind11.h>

#include <cstdint>
#include <memory>
#include <mutex>

#include "rlmeta/rpc/cc/queue_impl.h"
#include "rlmeta/rpc/cc/task.h"

namespace py = pybind11;

namespace rlmeta {
namespace rpc {

class ComputationQueue {
 public:
  ComputationQueue() = default;
  explicit ComputationQueue(int64_t capacity) : queue_impl_(capacity) {}

  virtual std::future<std::string> Put(const std::string& args,
                                       const std::string& kwargs) {
    std::shared_ptr<Task> task = std::make_shared<Task>(args, kwargs);
    queue_impl_.Put(task);
    return task->Future();
  }

  virtual std::shared_ptr<TaskBase> Get() {
    return queue_impl_.Get().value_or(nullptr);
  }

  virtual void Shutdown() { queue_impl_.Shutdown(); }

 protected:
  QueueImpl<std::shared_ptr<TaskBase>> queue_impl_;
};

class BatchedComputationQueue : public ComputationQueue {
 public:
  explicit BatchedComputationQueue(int64_t batch_size)
      : batch_size_(batch_size) {}
  BatchedComputationQueue(int64_t capacity, int64_t batch_size)
      : ComputationQueue(capacity), batch_size_(batch_size) {}

  std::future<std::string> Put(const std::string& args,
                               const std::string& kwargs) override;

  std::shared_ptr<TaskBase> Get() override;

  std::shared_ptr<TaskBase> GetFullBatch();

 protected:
  const int64_t batch_size_;
  std::shared_ptr<BatchedTask> cur_computation_ = nullptr;

  std::mutex mu_;
};

void DefineComputationQueue(py::module& m);
void DefineBatchedComputationQueue(py::module& m);

}  // namespace rpc
}  // namespace rlmeta
