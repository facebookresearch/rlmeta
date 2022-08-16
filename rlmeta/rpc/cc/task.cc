// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "rlmeta/rpc/cc/task.h"

#include <cassert>

namespace rlmeta {
namespace rpc {

void BatchedTask::SetReturnValue(const py::object& return_value) {
  NestedData ret = rpc_utils::PythonToNestedData(return_value);
  py::gil_scoped_release release;
  assert(ret.has_vec());
  assert(ret.vec_size() == batch_size_);
  for (int64_t i = 0; i < batch_size_; ++i) {
    promises_[i].set_value(std::move(*ret.mutable_vec()->mutable_data(i)));
  }
}

std::future<NestedData> BatchedTask::Add(const NestedData& args,
                                         const NestedData& kwargs) {
  assert(batch_size_ < capacity_);
  std::promise<NestedData>& p = promises_.emplace_back();
  *args_.mutable_vec()->add_data() = args;
  *kwargs_.mutable_vec()->add_data() = kwargs;
  ++batch_size_;
  num_to_wait_.DecrementCount();
  return p.get_future();
}

std::future<NestedData> BatchedTask::Add(NestedData&& args,
                                         NestedData&& kwargs) {
  assert(batch_size_ < capacity_);
  std::promise<NestedData>& p = promises_.emplace_back();
  *args_.mutable_vec()->add_data() = std::move(args);
  *kwargs_.mutable_vec()->add_data() = std::move(kwargs);
  ++batch_size_;
  num_to_wait_.DecrementCount();
  return p.get_future();
}

void DefineTask(py::module& m) {
  py::class_<Task, std::shared_ptr<Task>>(m, "Task")
      .def("args", &Task::Args)
      .def("kwargs", &Task::Kwargs)
      .def("set_return_value", &Task::SetReturnValue);
}

void DefineBatchedTask(py::module& m) {
  py::class_<BatchedTask, Task, std::shared_ptr<BatchedTask>>(m, "BatchedTask")
      .def("__len__", &BatchedTask::batch_size)
      .def_property_readonly("capacity", &BatchedTask::capacity)
      .def_property_readonly("batch_size", &BatchedTask::batch_size)
      .def("empty", &BatchedTask::Empty)
      .def("full", &BatchedTask::Full);
}

}  // namespace rpc
}  // namespace rlmeta
