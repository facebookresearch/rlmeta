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

// The Client class is adapted from gRPC C++ async client example.
// https://github.com/grpc/grpc/blob/2d4f3c56001cd1e1f85734b2f7c5ce5f2797c38a/examples/cpp/helloworld/greeter_async_client2.cc
//
// Copyright 2015 gRPC authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

  void AsyncCompleteRpc();

  std::string addr_;
  bool connected_ = false;

  std::shared_ptr<grpc::Channel> channel_;
  std::unique_ptr<rlmeta::rpc::Rpc::Stub> stub_;

  grpc::CompletionQueue cq_;
  std::unique_ptr<std::thread> thread_;
};

void DefineClient(py::module& m);

}  // namespace rpc
}  // namespace rlmeta
