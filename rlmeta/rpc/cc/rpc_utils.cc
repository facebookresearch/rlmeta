// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "rlmeta/rpc/cc/rpc_utils.h"

#include <pybind11/numpy.h>
#include <torch/extension.h>
#include <torch/python.h>
#include <torch/torch.h>

#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>

#include "rlmeta/cc/nested_utils.h"
#include "rlmeta/cc/torch_utils.h"
#include "rlmeta/rpc/cc/tensor_wrapper.h"

namespace rlmeta {
namespace rpc {

namespace rpc_utils {

TensorProto PythonToTensorProto(const py::object& obj) {
  if (py::isinstance<py::array>(obj)) {
    return TensorWrapper<py::array>(obj).TensorProto();
  }
  if (rlmeta::utils::IsTorchTensor(obj)) {
    return TensorWrapper<torch::Tensor>(obj).TensorProto();
  }
  return TensorProto();
}

py::object TensorProtoToPython(TensorProto&& tensor) {
  if (tensor.tensor_type() == TensorProto::NUMPY) {
    return TensorWrapper<py::array>(std::move(tensor)).Python();
  }
  if (tensor.tensor_type() == TensorProto::TORCH) {
    return TensorWrapper<torch::Tensor>(std::move(tensor)).Python();
  }
  return py::none();
}

SimpleData PythonToSimpleData(const py::object& obj) {
  SimpleData ret;
  if (obj.is_none()) {
    return ret;
  }
  if (py::isinstance<py::bool_>(obj)) {
    ret.set_bool_val(obj.cast<bool>());
  } else if (py::isinstance<py::int_>(obj)) {
    ret.set_int_val(obj.cast<int64_t>());
  } else if (py::isinstance<py::float_>(obj)) {
    ret.set_float_val(obj.cast<double>());
  } else if (py::isinstance<py::str>(obj)) {
    ret.set_str_val(obj.cast<std::string>());
  } else if (py::isinstance<py::bytes>(obj)) {
    ret.set_bytes_val(obj.cast<std::string>());
  } else if (py::isinstance<py::array>(obj)) {
    *ret.mutable_tensor_val() = PythonToTensorProto(obj);
  } else if (rlmeta::utils::IsTorchTensor(obj)) {
    *ret.mutable_tensor_val() = PythonToTensorProto(obj);
  }
  return ret;
}

py::object SimpleDataToPython(SimpleData&& proto) {
  if (proto.has_bool_val()) {
    return py::cast(proto.bool_val());
  }
  if (proto.has_int_val()) {
    return py::cast(proto.int_val());
  }
  if (proto.has_float_val()) {
    return py::cast(proto.float_val());
  }
  if (proto.has_str_val()) {
    return py::cast(proto.str_val());
  }
  if (proto.has_bytes_val()) {
    return py::bytes(proto.bytes_val());
  }
  if (proto.has_tensor_val()) {
    return TensorProtoToPython(std::move(*(proto.mutable_tensor_val())));
  }
  return py::none();
}

NestedData PythonToNestedData(const py::object& obj) {
  NestedData ret;
  if (obj.is_none()) {
    return ret;
  }

  if (py::isinstance<py::tuple>(obj)) {
    py::tuple src = py::reinterpret_borrow<py::tuple>(obj);
    auto* dst = ret.mutable_vec();
    for (const auto x : src) {
      *dst->add_data() =
          PythonToNestedData(py::reinterpret_borrow<py::object>(x));
    }
  } else if (py::isinstance<py::list>(obj)) {
    py::list src = py::reinterpret_borrow<py::list>(obj);
    auto* dst = ret.mutable_vec();
    for (const auto x : src) {
      *dst->add_data() =
          PythonToNestedData(py::reinterpret_borrow<py::object>(x));
    }
  } else if (py::isinstance<py::dict>(obj)) {
    py::dict src = py::reinterpret_borrow<py::dict>(obj);
    auto* dst = ret.mutable_map();
    const std::vector<std::string> keys = nested_utils::SortedKeys(src);
    for (const std::string& k : keys) {
      dst->mutable_data()->insert(
          {k, PythonToNestedData(
                  py::reinterpret_borrow<py::object>(src[py::str(k)]))});
    }
  } else {
    *ret.mutable_val() = PythonToSimpleData(obj);
  }

  return ret;
}

py::object NestedDataToPython(NestedData&& proto) {
  if (proto.has_val()) {
    return SimpleDataToPython(std::move(*proto.mutable_val()));
  }
  if (proto.has_vec()) {
    auto* src = proto.mutable_vec();
    const int64_t n = src->data_size();
    py::tuple ret(n);
    for (int64_t i = 0; i < n; ++i) {
      ret[i] = NestedDataToPython(std::move(*(src->mutable_data(i))));
    }
    return ret;
  }
  if (proto.has_map()) {
    auto* src = proto.mutable_map()->mutable_data();
    py::dict ret;
    const std::vector<std::string> keys = nested_utils::SortedKeys(*src);
    for (const std::string& k : keys) {
      ret[py::str(k)] = NestedDataToPython(std::move(src->at(k)));
    }
    return ret;
  }
  return py::none();
}

std::string Dumps(const py::object& obj) {
  const NestedData src = PythonToNestedData(obj);
  py::gil_scoped_release release;
  return src.SerializeAsString();
}

py::object Loads(const std::string& src) {
  NestedData dst;
  {
    py::gil_scoped_release release;
    dst.ParseFromString(src);
  }
  return NestedDataToPython(std::move(dst));
}

}  // namespace rpc_utils

void DefineRpcUtils(py::module& m) {
  m.def("dumps", [](const py::object& obj) {
     return py::bytes(rpc_utils::Dumps(obj));
   }).def("loads", &rpc_utils::Loads);
}

}  // namespace rpc
}  // namespace rlmeta
