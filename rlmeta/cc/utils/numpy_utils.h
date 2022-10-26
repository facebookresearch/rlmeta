// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <pybind11/numpy.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

namespace py = pybind11;

namespace rlmeta {
namespace utils {

template <typename T>
std::vector<int64_t> NumpyArrayShape(const py::array_t<T>& arr) {
  const int64_t ndim = arr.ndim();
  std::vector<int64_t> shape(ndim);
  for (int64_t i = 0; i < ndim; ++i) {
    shape[i] = static_cast<int64_t>(arr.shape(i));
  }
  return shape;
}

template <typename T_SRC, typename T_DST = T_SRC>
py::array_t<T_DST> NumpyEmptyLike(const py::array_t<T_SRC>& src) {
  const std::vector<int64_t> shape = NumpyArrayShape(src);
  return py::array_t<T_DST>(shape);
}

template <typename T>
py::array_t<T> AsNumpyArray(const std::vector<T>& vec) {
  py::array_t<T> arr(vec.size());
  std::memcpy(arr.mutable_data(), vec.data(), vec.size() * sizeof(T));
  return arr;
}

// Move a std::vector to numpy array without copy.
// https://github.com/ssciwr/pybind11-numpy-example/blob/main/python/pybind11-numpy-example_python.cpp
template <typename T>
py::array_t<T> AsNumpyArray(std::vector<T>&& vec) {
  const int64_t size = vec.size();
  const T* data = vec.data();
  std::unique_ptr<std::vector<T>> vec_ptr =
      std::make_unique<std::vector<T>>(std::move(vec));
  auto capsule = py::capsule(vec_ptr.get(), [](void* p) {
    std::unique_ptr<std::vector<T>>(reinterpret_cast<std::vector<T>*>(p));
  });
  vec_ptr.release();
  return py::array(size, data, capsule);
}

}  // namespace utils
}  // namespace rlmeta
