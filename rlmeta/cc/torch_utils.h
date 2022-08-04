// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/extension.h>
#include <torch/python.h>
#include <torch/torch.h>

#include <cstdint>

namespace rlmeta {
namespace utils {

template <typename T>
struct TorchDataType;

template <>
struct TorchDataType<bool> {
  static constexpr torch::ScalarType value = torch::kBool;
};

template <>
struct TorchDataType<uint8_t> {
  static constexpr torch::ScalarType value = torch::kUInt8;
};

template <>
struct TorchDataType<int8_t> {
  static constexpr torch::ScalarType value = torch::kInt8;
};

template <>
struct TorchDataType<int16_t> {
  static constexpr torch::ScalarType value = torch::kInt16;
};

template <>
struct TorchDataType<int32_t> {
  static constexpr torch::ScalarType value = torch::kInt32;
};

template <>
struct TorchDataType<int64_t> {
  static constexpr torch::ScalarType value = torch::kInt64;
};

template <>
struct TorchDataType<float> {
  static constexpr torch::ScalarType value = torch::kFloat;
};

template <>
struct TorchDataType<double> {
  static constexpr torch::ScalarType value = torch::kDouble;
};

inline bool IsTorchTensor(const py::object& obj) {
  return THPVariable_Check(obj.ptr());
}

inline torch::Tensor PyObjectToTorchTensor(const py::object& obj) {
  return THPVariable_Unpack(obj.ptr());
}

inline py::object TorchTensorToPyObject(const torch::Tensor& tensor) {
  return py::reinterpret_steal<py::object>(THPVariable_Wrap(tensor));
}

}  // namespace utils
}  // namespace rlmeta
