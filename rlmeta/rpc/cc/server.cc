// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "rlmeta/rpc/cc/server.h"

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/health_check_service_interface.h>

#include <future>
#include <mutex>

namespace rlmeta {
namespace rpc {

grpc::Status ServiceImpl::RemoteCall(grpc::ServerContext* /*context*/,
                                     const RpcRequest* request,
                                     RpcResponse* response) {
  auto& func = functions_.at(request->function());
  response->set_return_value(func(request->args(), request->kwargs()));
  return grpc::Status::OK;
}

void Server::Start() {
  grpc::EnableDefaultHealthCheckService(true);
  grpc::reflection::InitProtoReflectionServerBuilderPlugin();
  grpc::ServerBuilder builder;
  builder.AddListeningPort(addr_, grpc::InsecureServerCredentials());
  builder.RegisterService(&service_);
  server_ = builder.BuildAndStart();
}

void Server::Stop() {
  server_->Shutdown();
  server_->Wait();
}

std::shared_ptr<ComputationQueue> Server::RegisterQueue(
    const std::string& func_name, int64_t batch_size) {
  std::shared_ptr<ComputationQueue> ret = nullptr;
  if (batch_size == 0) {
    ret = std::make_shared<ComputationQueue>();
  } else {
    ret = std::make_shared<BatchedComputationQueue>(batch_size);
  }
  service_.Register(func_name, [que = ret](const std::string& args,
                                           const std::string& kwargs) {
    std::future<std::string> ret = que->Put(args, kwargs);
    return ret.get();
  });
  return ret;
}

void DefineServer(py::module& m) {
  py::class_<Server, std::shared_ptr<Server>>(m, "Server")
      .def(py::init<const std::string&>())
      .def("start", &Server::Start)
      .def("stop", &Server::Stop, py::call_guard<py::gil_scoped_release>())
      .def("register_queue", &Server::RegisterQueue, py::arg("func_name"),
           py::arg("batch_size") = 0);
}

}  // namespace rpc
}  // namespace rlmeta
