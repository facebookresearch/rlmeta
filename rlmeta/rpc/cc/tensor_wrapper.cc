// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "rlmeta/rpc/cc/tensor_wrapper.h"

#include <torch/torch.h>

#include <string>

#include "rlmeta/cc/torch_utils.h"

namespace rlmeta {
namespace rpc {

template <>
void TensorWrapper<py::array>::FromPython(const py::object& obj) {
  tensor_ = obj.cast<py::array>();
}

template <>
void TensorWrapper<py::array>::FromTensorProto(
    rlmeta::rpc::TensorProto&& proto) {
  assert(tensor.tensor_type() == TensorProto::NUMPY);
  const std::vector<int64_t> shape(proto.shape().cbegin(),
                                   proto.shape().cend());
  const void* data = proto.data().data();
  std::unique_ptr<std::string> data_ptr(proto.release_data());
  auto capsule = py::capsule(data_ptr.get(), [](void* p) {
    std::unique_ptr<std::string>(reinterpret_cast<std::string*>(p));
  });
  data_ptr.release();
  tensor_ = py::array(py::dtype(proto.dtype()), shape, data, capsule);
}

template <>
py::object TensorWrapper<py::array>::Python() {
  return std::move(tensor_);
}

template <>
rlmeta::rpc::TensorProto TensorWrapper<py::array>::TensorProto() {
  rlmeta::rpc::TensorProto ret;
  ret.set_tensor_type(TensorProto::NUMPY);
  ret.set_dtype(tensor_.dtype().num());
  ret.mutable_shape()->Assign(tensor_.shape(),
                              tensor_.shape() + tensor_.ndim());
  ret.set_data(tensor_.data(), tensor_.nbytes());
  return ret;
}

template <>
void TensorWrapper<torch::Tensor>::FromPython(const py::object& obj) {
  tensor_ = rlmeta::utils::PyObjectToTorchTensor(obj).contiguous();
}

template <>
void TensorWrapper<torch::Tensor>::FromTensorProto(
    rlmeta::rpc::TensorProto&& proto) {
  assert(proto.tensor_type() == Tensor::TORCH);
  const std::vector<int64_t> shape(proto.shape().cbegin(),
                                   proto.shape().cend());
  std::string* data = proto.release_data();
  tensor_ = at::from_blob(
      data->data(), shape, /*deleter=*/[data](void* /*p*/) { delete data; },
      static_cast<at::ScalarType>(proto.dtype()));
}

template <>
py::object TensorWrapper<torch::Tensor>::Python() {
  return rlmeta::utils::TorchTensorToPyObject(std::move(tensor_));
}

template <>
rlmeta::rpc::TensorProto TensorWrapper<torch::Tensor>::TensorProto() {
  rlmeta::rpc::TensorProto ret;
  ret.set_tensor_type(TensorProto::TORCH);
  ret.set_dtype(static_cast<int32_t>(tensor_.scalar_type()));
  ret.mutable_shape()->Assign(tensor_.sizes().cbegin(), tensor_.sizes().cend());
  ret.set_data(tensor_.data_ptr(), tensor_.nbytes());
  return ret;
}

}  // namespace rpc
}  // namespace rlmeta
