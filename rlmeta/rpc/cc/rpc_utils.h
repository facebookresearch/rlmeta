// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <pybind11/pybind11.h>

#include <string>

#include "rpc.pb.h"

namespace py = pybind11;

namespace rlmeta {
namespace rpc {

namespace rpc_utils {

TensorProto PythonToTensorProto(const py::object& obj);
py::object TensorProtoToPython(TensorProto&& tensor);

SimpleData PythonToSimpleData(const py::object& obj);
py::object SimpleDataToPython(SimpleData&& proto);

NestedData PythonToNestedData(const py::object& obj);
py::object NestedDataToPython(NestedData&& proto);

std::string Dumps(const py::object& obj);
py::object Loads(const std::string& src);

}  // namespace rpc_utils

void DefineRpcUtils(py::module& m);

}  // namespace rpc
}  // namespace rlmeta
