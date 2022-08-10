// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <grpc/support/log.h>
#include <grpcpp/grpcpp.h>
#include <pybind11/pybind11.h>

#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "rlmeta/rpc/cc/computation_queue.h"
#include "rlmeta/rpc/cc/task.h"
#include "rpc.grpc.pb.h"
#include "rpc.pb.h"

namespace py = pybind11;

namespace rlmeta {
namespace rpc {

using PyFunc = std::function<NestedData(const NestedData&, const NestedData&)>;
using PyFuncRvalue = std::function<NestedData(NestedData&&, NestedData&&)>;
using PyFuncDict =
    std::unordered_map<std::string, std::pair<PyFunc, PyFuncRvalue>>;

class ServiceImpl final : public Rpc::Service {
 public:
  grpc::Status Register(const std::string& func_name, PyFunc&& func_impl,
                        PyFuncRvalue&& func_rvalue_impl) {
    functions_.emplace(func_name, std::make_pair(std::move(func_impl),
                                                 std::move(func_rvalue_impl)));
    return grpc::Status::OK;
  }

 private:
  grpc::Status RemoteCall(grpc::ServerContext* context,
                          const RpcRequest* request,
                          RpcResponse* response) override;

  grpc::Status PyRemoteCall(grpc::ServerContext* context,
                            const PyRpcRequest* request,
                            PyRpcResponse* response) override;

  PyFuncDict functions_;
};

class Server {
 public:
  explicit Server(const std::string& addr) : addr_(addr) {}
  ~Server() { Stop(); }

  const std::string& addr() const { return addr_; }

  void Start();
  void Stop();

  std::shared_ptr<ComputationQueue> RegisterQueue(const std::string& func_name,
                                                  int64_t batch_size = 0);

 protected:
  void ServePyFuncQueue(const std::string& func_name);

  const std::string addr_;
  std::unique_ptr<grpc::Server> server_;
  ServiceImpl service_;
};

void DefineServer(py::module& m);

}  // namespace rpc
}  // namespace rlmeta
