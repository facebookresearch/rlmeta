// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <pybind11/numpy.h>

#include "rpc.pb.h"

namespace py = pybind11;

namespace rlmeta {
namespace rpc {

class TensorWrapperBase {
 public:
  virtual void FromPython(const py::object& obj) = 0;
  virtual void FromTensorProto(rlmeta::rpc::TensorProto&& proto) = 0;

  virtual py::object Python() = 0;
  virtual rlmeta::rpc::TensorProto TensorProto() = 0;
};

template <class TensorType>
class TensorWrapper : public TensorWrapperBase {
 public:
  TensorWrapper(const py::object& obj) { FromPython(obj); }
  TensorWrapper(rlmeta::rpc::TensorProto&& proto) {
    FromTensorProto(std::move(proto));
  }

  void FromPython(const py::object& obj) override;
  void FromTensorProto(rlmeta::rpc::TensorProto&& proto) override;

  py::object Python() override;
  rlmeta::rpc::TensorProto TensorProto() override;

 protected:
  TensorType tensor_;
};

}  // namespace rpc
}  // namespace rlmeta
