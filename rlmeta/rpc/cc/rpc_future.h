// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <pybind11/pybind11.h>

#include <future>

#include "rpc.pb.h"

namespace py = pybind11;

namespace rlmeta {
namespace rpc {

class RpcFuture {
 public:
  RpcFuture(std::future<NestedData>&& fut) : future_(std::move(fut)) {}

  bool Valid() const { return future_.valid() && valid_; }

  py::object Get();

  void Wait() { future_.wait(); }

 protected:
  std::future<NestedData> future_;
  bool valid_ = true;
};

void DefineRpcFuture(py::module& m);

}  // namespace rpc
}  // namespace rlmeta
