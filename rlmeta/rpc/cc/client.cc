// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "rlmeta/rpc/cc/client.h"

#include <cassert>
#include <chrono>

#include "rlmeta/rpc/cc/rpc_utils.h"

namespace rlmeta {
namespace rpc {

void Client::Connect(const std::string& addr, int64_t timeout) {
  if (connected_) {
    Disconnect();
  }

  grpc::ChannelArguments ch_args;
  ch_args.SetMaxSendMessageSize(-1);     // Unlimited
  ch_args.SetMaxReceiveMessageSize(-1);  // Unlimited
  channel_ = grpc::CreateCustomChannel(addr, grpc::InsecureChannelCredentials(),
                                       ch_args);
  stub_ = Rpc::NewStub(channel_);

  const auto deadline =
      std::chrono::system_clock::now() + std::chrono::seconds(timeout);
  if (!channel_->WaitForConnected(deadline)) {
    std::cerr << "[Client::connect] timeout" << std::endl;
  }
  connected_ = true;
}

void Client::Disconnect() {
  if (connected_) {
    connected_ = false;
  }
}

py::object Client::Rpc(const std::string& func, const py::args& args,
                       const py::kwargs& kwargs) {
  assert(connected_);
  RpcRequest request;
  request.set_function(func);
  *request.mutable_args() = rpc_utils::PythonToNestedData(args);
  *request.mutable_kwargs() = rpc_utils::PythonToNestedData(kwargs);

  NestedData ret;
  {
    py::gil_scoped_release release;
    ret = RpcImpl(std::move(request));
  }

  return rpc_utils::NestedDataToPython(std::move(ret));
}

rlmeta::rpc::RpcFuture Client::RpcFuture(const std::string& func,
                                         const py::args& args,
                                         const py::kwargs& kwargs) {
  assert(connected_);
  RpcRequest request;
  request.set_function(func);
  *request.mutable_args() = rpc_utils::PythonToNestedData(args);
  *request.mutable_kwargs() = rpc_utils::PythonToNestedData(kwargs);

  py::gil_scoped_release release;
  rlmeta::rpc::RpcFuture fut =
      std::async(&Client::RpcImpl, this, std::move(request));
  return fut;
}

NestedData Client::RpcImpl(RpcRequest&& request) {
  RpcResponse response;
  grpc::ClientContext context;
  grpc::Status status = stub_->RemoteCall(&context, request, &response);
  assert(status.ok());
  NestedData ret = std::move(*response.mutable_return_value());
  return ret;
}

void DefineClient(py::module& m) {
  py::class_<Client, std::shared_ptr<Client>>(m, "Client")
      .def(py::init<>())
      .def_property_readonly("addr", &Client::addr)
      .def_property_readonly("connected", &Client::connected)
      .def("connect", &Client::Connect, py::arg("addr"),
           py::arg("timeout") = 60, py::call_guard<py::gil_scoped_release>())
      .def("disconnect", &Client::Disconnect,
           py::call_guard<py::gil_scoped_release>())
      .def("rpc", &Client::Rpc)
      .def("rpc_future", &Client::RpcFuture);
}

}  // namespace rpc
}  // namespace rlmeta
