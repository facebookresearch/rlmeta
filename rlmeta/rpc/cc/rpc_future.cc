// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "rlmeta/rpc/cc/rpc_future.h"

#include <memory>

#include "rlmeta/rpc/cc/rpc_utils.h"

namespace rlmeta {
namespace rpc {

py::object RpcFuture::Get() {
  if (!Valid()) {
    return py::none();
  }
  py::object ret = rpc_utils::NestedDataToPython(std::move(future_.get()));
  valid_ = false;
  return ret;
}

void DefineRpcFuture(py::module& m) {
  py::class_<RpcFuture, std::shared_ptr<RpcFuture>>(m, "RpcFuture")
      .def("valid", &RpcFuture::Valid)
      .def("get", &RpcFuture::Get)
      .def("wait", &RpcFuture::Wait);
}

}  // namespace rpc
}  // namespace rlmeta
