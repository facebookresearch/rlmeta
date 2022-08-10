// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <grpcpp/grpcpp.h>
#include <pybind11/pybind11.h>

#include <future>
#include <memory>
#include <string>
#include <thread>

#include "rlmeta/rpc/cc/rpc_future.h"
#include "rpc.grpc.pb.h"
#include "rpc.pb.h"

namespace py = pybind11;

namespace rlmeta {
namespace rpc {

class Client {
 public:
  Client() = default;
  ~Client() { Disconnect(); }

  const std::string& addr() const { return addr_; }
  bool connected() const { return connected_; }

  void Connect(const std::string& addr, int64_t timeout = 60);
  void Disconnect();

  py::object Rpc(const std::string& func, const py::args& args,
                 const py::kwargs& kwargs);
  rlmeta::rpc::RpcFuture RpcFuture(const std::string& func,
                                   const py::args& args,
                                   const py::kwargs& kwargs);

 protected:
  struct AsyncClientCall {
    RpcResponse response;
    grpc::ClientContext context;
    grpc::Status status;
    std::promise<NestedData> promise;
    std::unique_ptr<grpc::ClientAsyncResponseReader<RpcResponse>>
        response_reader;
  };

  NestedData RpcImpl(RpcRequest&& request);

  std::string addr_;
  bool connected_ = false;

  std::shared_ptr<grpc::Channel> channel_;
  std::unique_ptr<rlmeta::rpc::Rpc::Stub> stub_;
};

void DefineClient(py::module& m);

}  // namespace rpc
}  // namespace rlmeta
