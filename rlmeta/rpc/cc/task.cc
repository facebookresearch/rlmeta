// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "rlmeta/rpc/cc/task.h"

#include <cassert>

namespace rlmeta {
namespace rpc {

py::object BatchedTask::Args() {
  const int64_t batch_size = batch_.size();
  py::tuple ret(batch_size);
  for (int64_t i = 0; i < batch_size; ++i) {
    ret[i] = batch_[i].Args();
  }
  return ret;
}

py::object BatchedTask::Kwargs() {
  const int64_t batch_size = batch_.size();
  py::tuple ret(batch_size);
  for (int64_t i = 0; i < batch_size; ++i) {
    ret[i] = batch_[i].Kwargs();
  }
  return ret;
}

void BatchedTask::SetReturnValue(py::object&& return_value) {
  const int64_t batch_size = batch_.size();
  py::tuple rets = py::reinterpret_borrow<py::tuple>(return_value);
  assert(rets.size() == batch_size);
  for (int64_t i = 0; i < batch_size; ++i) {
    batch_[i].SetReturnValue(std::move(rets[i]));
  }
}

std::future<std::string> BatchedTask::Add(const std::string& args,
                                          const std::string& kwargs) {
  const int64_t batch_size = batch_.size();
  assert(batch_size < capacity_);
  auto& task = batch_.emplace_back(args, kwargs);
  num_to_wait_.DecrementCount();
  return task.Future();
}

void DefineTaskBase(py::module& m) {
  py::class_<TaskBase, PyTaskBase, std::shared_ptr<TaskBase>>(m, "TaskBase")
      .def("args", &TaskBase::Args)
      .def("kwargs", &TaskBase::Kwargs)
      .def("set_return_value", &TaskBase::SetReturnValue);
}

void DefineTask(py::module& m) {
  py::class_<Task, TaskBase, std::shared_ptr<Task>>(m, "Task");
}

void DefineBatchedTask(py::module& m) {
  py::class_<BatchedTask, TaskBase, std::shared_ptr<BatchedTask>>(m,
                                                                  "BatchedTask")
      .def("__len__", &BatchedTask::BatchSize)
      .def_property_readonly("capacity", &BatchedTask::capacity)
      .def_property_readonly("batch_size", &BatchedTask::BatchSize)
      .def("empty", &BatchedTask::Empty)
      .def("full", &BatchedTask::Full);
}

}  // namespace rpc
}  // namespace rlmeta
