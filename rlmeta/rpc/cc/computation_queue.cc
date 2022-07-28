// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "rlmeta/rpc/cc/computation_queue.h"

namespace rlmeta {
namespace rpc {

std::future<std::string> BatchedComputationQueue::Put(
    const std::string& args, const std::string& kwargs) {
  std::scoped_lock lk(mu_);
  if (cur_computation_ == nullptr) {
    cur_computation_ = std::make_shared<BatchedTask>(batch_size_);
    queue_impl_.Put(cur_computation_);
  }
  std::future<std::string> ret = cur_computation_->Add(args, kwargs);
  if (cur_computation_->Full()) {
    cur_computation_.reset();
  }
  return ret;
}

std::shared_ptr<TaskBase> BatchedComputationQueue::Get() {
  std::shared_ptr<TaskBase> ret = queue_impl_.Get().value_or(nullptr);
  if (ret != nullptr) {
    std::scoped_lock lk(mu_);
    if (!dynamic_cast<BatchedTask*>(ret.get())->Full()) {
      cur_computation_.reset();
    }
  }
  return ret;
}

std::shared_ptr<TaskBase> BatchedComputationQueue::GetFullBatch() {
  std::shared_ptr<TaskBase> ret = queue_impl_.Get().value_or(nullptr);
  if (ret != nullptr) {
    dynamic_cast<BatchedTask*>(ret.get())->Wait();
  }
  return ret;
}

void DefineComputationQueue(py::module& m) {
  py::class_<ComputationQueue, std::shared_ptr<ComputationQueue>>(
      m, "ComputationQueue")
      .def(py::init<>())
      .def("get", &ComputationQueue::Get,
           py::call_guard<py::gil_scoped_release>())
      .def("shutdown", &ComputationQueue::Shutdown,
           py::call_guard<py::gil_scoped_release>());
}

void DefineBatchedComputationQueue(py::module& m) {
  py::class_<BatchedComputationQueue, ComputationQueue,
             std::shared_ptr<BatchedComputationQueue>>(
      m, "BatchedComputationQueue")
      .def(py::init<int64_t>())
      .def("get_full_batch", &BatchedComputationQueue::GetFullBatch,
           py::call_guard<py::gil_scoped_release>());
}

}  // namespace rpc
}  // namespace rlmeta
