// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include "rlmeta/cc/utils/numpy_utils.h"
#include "rlmeta/cc/utils/segment_tree.h"
#include "rlmeta/cc/utils/torch_utils.h"

namespace py = pybind11;

namespace rlmeta {

namespace {

template <typename T>
void SumSegmentTreeAt(const SumSegmentTree<T>& sum_tree, int64_t n,
                      const int64_t* index, T* value) {
  for (int64_t i = 0; i < n; ++i) {
    value[i] = sum_tree.At(index[i]);
  }
}

template <typename T>
py::array_t<T> SumSegmentTreeAt(const SumSegmentTree<T>& sum_tree,
                                const py::array_t<int64_t>& index) {
  py::array_t<T> value = utils::NumpyEmptyLike<int64_t, T>(index);
  SumSegmentTreeAt<T>(sum_tree, index.size(), index.data(),
                      value.mutable_data());
  return value;
}

template <typename T>
torch::Tensor SumSegmentTreeAt(const SumSegmentTree<T>& sum_tree,
                               const torch::Tensor& index) {
  assert(index.dtype() == torch::kInt64);
  const torch::Tensor index_contiguous = index.contiguous();
  torch::Tensor value =
      torch::empty_like(index_contiguous, utils::TorchDataType<T>::value);
  SumSegmentTreeAt<T>(sum_tree, index_contiguous.numel(),
                      index_contiguous.data_ptr<int64_t>(),
                      value.data_ptr<T>());
  return value;
}

template <typename T>
void SumSegmentTreeUpdate(int64_t n, const int64_t* index, T value,
                          SumSegmentTree<T>& sum_tree) {
  for (int64_t i = 0; i < n; ++i) {
    sum_tree.Update(index[i], value);
  }
}

template <typename T>
void SumSegmentTreeUpdate(int64_t n, const int64_t* index, const T* value,
                          SumSegmentTree<T>& sum_tree) {
  for (int64_t i = 0; i < n; ++i) {
    sum_tree.Update(index[i], value[i]);
  }
}

template <typename T>
void SumSegmentTreeUpdate(int64_t n, const int64_t* index, const T* value,
                          const bool* mask, SumSegmentTree<T>& sum_tree) {
  for (int64_t i = 0; i < n; ++i) {
    if (mask[i]) {
      sum_tree.Update(index[i], value[i]);
    }
  }
}

template <typename T>
void SumSegmentTreeUpdate(const py::array_t<int64_t>& index, T value,
                          SumSegmentTree<T>& sum_tree) {
  SumSegmentTreeUpdate<T>(index.size(), index.data(), value, sum_tree);
}

template <typename T>
void SumSegmentTreeUpdate(const py::array_t<int64_t>& index,
                          const py::array_t<T>& value,
                          SumSegmentTree<T>& sum_tree) {
  assert(index.size() == value.size());
  SumSegmentTreeUpdate<T>(index.size(), index.data(), value.data(), sum_tree);
}

template <typename T>
void SumSegmentTreeUpdate(const py::array_t<int64_t>& index,
                          const py::array_t<T>& value,
                          const py::array_t<bool>& mask,
                          SumSegmentTree<T>& sum_tree) {
  assert(index.size() == value.size());
  assert(index.size() == mask.size());
  SumSegmentTreeUpdate<T>(index.size(), index.data(), value.data(), mask.data(),
                          sum_tree);
}

template <typename T>
void SumSegmentTreeUpdate(const torch::Tensor& index, T value,
                          SumSegmentTree<T>& sum_tree) {
  assert(index.dtype() == torch::kInt64);
  const torch::Tensor index_contiguous = index.contiguous();
  SumSegmentTreeUpdate<T>(index_contiguous.numel(),
                          index_contiguous.data_ptr<int64_t>(), value,
                          sum_tree);
}

template <typename T>
void SumSegmentTreeUpdate(const torch::Tensor& index,
                          const torch::Tensor& value,
                          SumSegmentTree<T>& sum_tree) {
  assert(index.dtype() == torch::kInt64);
  assert(value.dtype() == utils::TorchDataType<T>::value);
  const torch::Tensor index_contiguous = index.contiguous();
  const torch::Tensor value_contiguous = value.contiguous();
  SumSegmentTreeUpdate<T>(index_contiguous.numel(),
                          index_contiguous.data_ptr<int64_t>(),
                          value_contiguous.data_ptr<T>(), sum_tree);
}

template <typename T>
void SumSegmentTreeUpdate(const torch::Tensor& index,
                          const torch::Tensor& value, const torch::Tensor& mask,
                          SumSegmentTree<T>& sum_tree) {
  assert(index.dtype() == torch::kInt64);
  assert(value.dtype() == utils::TorchDataType<T>::value);
  assert(mask.dtype() == torch::kBool);
  const torch::Tensor index_contiguous = index.contiguous();
  const torch::Tensor value_contiguous = value.contiguous();
  const torch::Tensor mask_contiguous = mask.contiguous();
  SumSegmentTreeUpdate<T>(index_contiguous.numel(),
                          index_contiguous.data_ptr<int64_t>(),
                          value_contiguous.data_ptr<T>(),
                          mask_contiguous.data_ptr<bool>(), sum_tree);
}

template <typename T>
void SumSegmentTreeQuery(const SumSegmentTree<T>& sum_tree, int64_t n,
                         const int64_t* l, const int64_t* r, T* result) {
  for (int64_t i = 0; i < n; ++i) {
    result[i] = sum_tree.Query(l[i], r[i]);
  }
}

template <typename T>
py::array_t<T> SumSegmentTreeQuery(const SumSegmentTree<T>& sum_tree,
                                   const py::array_t<int64_t>& l,
                                   const py::array_t<int64_t>& r) {
  assert(l.size() == r.size());
  py::array_t<T> ret = utils::NumpyEmptyLike<int64_t, T>(l);
  SumSegmentTreeQuery<T>(sum_tree, l.size(), l.data(), r.data(),
                         ret.mutable_data());
  return ret;
}

template <typename T>
torch::Tensor SumSegmentTreeQuery(const SumSegmentTree<T>& sum_tree,
                                  const torch::Tensor& l,
                                  const torch::Tensor& r) {
  assert(l.dtype() == torch::kInt64);
  assert(r.dtype() == torch::kInt64);
  assert(l.sizes() == r.sizes());
  const torch::Tensor l_contiguous = l.contiguous();
  const torch::Tensor r_contiguous = r.contiguous();
  torch::Tensor ret =
      torch::empty_like(l_contiguous, utils::TorchDataType<T>::value);
  SumSegmentTreeQuery<T>(sum_tree, l_contiguous.numel(),
                         l_contiguous.data_ptr<int64_t>(),
                         r_contiguous.data_ptr<int64_t>(), ret.data_ptr<T>());
  return ret;
}

template <typename T>
void SumSegmentTreeScanLowerBound(const SumSegmentTree<T>& sum_tree, int64_t n,
                                  const T* value, int64_t* index) {
  for (int64_t i = 0; i < n; ++i) {
    index[i] = sum_tree.ScanLowerBound(value[i]);
  }
}

template <typename T>
py::array_t<int64_t> SumSegmentTreeScanLowerBound(
    const SumSegmentTree<T>& sum_tree, const py::array_t<T>& value) {
  py::array_t<int64_t> index = utils::NumpyEmptyLike<T, int64_t>(value);
  SumSegmentTreeScanLowerBound<T>(sum_tree, value.size(), value.data(),
                                  index.mutable_data());
  return index;
}

template <typename T>
torch::Tensor SumSegmentTreeScanLowerBound(const SumSegmentTree<T>& sum_tree,
                                           const torch::Tensor& value) {
  assert(value.dtype() == utils::TorchDataType<T>::value);
  const torch::Tensor value_contiguous = value.contiguous();
  torch::Tensor index = torch::empty_like(value_contiguous, torch::kInt64);
  SumSegmentTreeScanLowerBound<T>(sum_tree, value_contiguous.numel(),
                                  value_contiguous.data_ptr<T>(),
                                  index.data_ptr<int64_t>());
  return index;
}

}  // namespace

template <typename T>
void DefineSumSegmentTree(const std::string& type, py::module& m) {
  const std::string pyclass = "SumSegmentTree" + type;
  py::class_<SumSegmentTree<T>, std::shared_ptr<SumSegmentTree<T>>>(
      m, pyclass.c_str())
      .def(py::init<int64_t>())
      .def_property_readonly("size", &SumSegmentTree<T>::size)
      .def_property_readonly("capacity", &SumSegmentTree<T>::capacity)
      .def_property_readonly("identity_element",
                             &SumSegmentTree<T>::identity_element)
      .def("__len__", &SumSegmentTree<T>::size)
      .def("__getitem__", &SumSegmentTree<T>::At)
      .def("__getitem__",
           [](const SumSegmentTree<T>& sum_tree, const py::array_t<T>& index) {
             return SumSegmentTreeAt<T>(sum_tree, index);
           })
      .def("__getitem__",
           [](const SumSegmentTree<T>& sum_tree, const torch::Tensor& index) {
             return SumSegmentTreeAt<T>(sum_tree, index);
           })
      .def("__setitem__", &SumSegmentTree<T>::Update)
      .def("__setitem__",
           [](SumSegmentTree<T>& sum_tree, const py::array_t<int64_t>& index,
              T value) {
             return SumSegmentTreeUpdate<T>(index, value, sum_tree);
           })
      .def("__setitem__",
           [](SumSegmentTree<T>& sum_tree, const py::array_t<int64_t>& index,
              const py::array_t<T>& value) {
             return SumSegmentTreeUpdate<T>(index, value, sum_tree);
           })
      .def(
          "__setitem__",
          [](SumSegmentTree<T>& sum_tree, const torch::Tensor& index, T value) {
            return SumSegmentTreeUpdate<T>(index, value, sum_tree);
          })
      .def("__setitem__",
           [](SumSegmentTree<T>& sum_tree, const torch::Tensor& index,
              const torch::Tensor& value) {
             return SumSegmentTreeUpdate<T>(index, value, sum_tree);
           })
      .def("at", &SumSegmentTree<T>::At)
      .def("at",
           [](const SumSegmentTree<T>& sum_tree, const py::array_t<T>& index) {
             return SumSegmentTreeAt<T>(sum_tree, index);
           })
      .def("at",
           [](const SumSegmentTree<T>& sum_tree, const torch::Tensor& index) {
             return SumSegmentTreeAt<T>(sum_tree, index);
           })
      .def("update", &SumSegmentTree<T>::Update)
      .def("update",
           [](SumSegmentTree<T>& sum_tree, const py::array_t<int64_t>& index,
              T value) {
             return SumSegmentTreeUpdate<T>(index, value, sum_tree);
           })
      .def("update",
           [](SumSegmentTree<T>& sum_tree, const py::array_t<int64_t>& index,
              const py::array_t<T>& value) {
             return SumSegmentTreeUpdate<T>(index, value, sum_tree);
           })
      .def("update",
           [](SumSegmentTree<T>& sum_tree, const py::array_t<int64_t>& index,
              const py::array_t<T>& value, const py::array_t<bool>& mask) {
             return SumSegmentTreeUpdate<T>(index, value, mask, sum_tree);
           })
      .def(
          "update",
          [](SumSegmentTree<T>& sum_tree, const torch::Tensor& index, T value) {
            return SumSegmentTreeUpdate<T>(index, value, sum_tree);
          })
      .def("update",
           [](SumSegmentTree<T>& sum_tree, const torch::Tensor& index,
              const torch::Tensor& value) {
             return SumSegmentTreeUpdate<T>(index, value, sum_tree);
           })
      .def("update",
           [](SumSegmentTree<T>& sum_tree, const torch::Tensor& index,
              const torch::Tensor& value, const torch::Tensor& mask) {
             return SumSegmentTreeUpdate<T>(index, value, mask, sum_tree);
           })
      .def("query", &SumSegmentTree<T>::Query)
      .def("query",
           [](const SumSegmentTree<T>& sum_tree, const py::array_t<int64_t>& l,
              const py::array_t<int64_t>& r) {
             return SumSegmentTreeQuery<T>(sum_tree, l, r);
           })
      .def("query",
           [](const SumSegmentTree<T>& sum_tree, const torch::Tensor& l,
              const torch::Tensor& r) {
             return SumSegmentTreeQuery<T>(sum_tree, l, r);
           })
      .def("scan_lower_bound", &SumSegmentTree<T>::ScanLowerBound)
      .def("scan_lower_bound",
           [](const SumSegmentTree<T>& sum_tree, const py::array_t<T>& value) {
             return SumSegmentTreeScanLowerBound<T>(sum_tree, value);
           })
      .def("scan_lower_bound",
           [](const SumSegmentTree<T>& sum_tree, const torch::Tensor& value) {
             return SumSegmentTreeScanLowerBound<T>(sum_tree, value);
           })
      .def(py::pickle(
          [](const SumSegmentTree<T>& sum_tree) {
            const int64_t n = sum_tree.size();
            py::array_t<T> ret(n);
            T* ret_data = ret.mutable_data();
            for (int64_t i = 0; i < n; ++i) {
              ret_data[i] = sum_tree.At(i);
            }
            return ret;
          },
          [](const py::array_t<T>& arr) {
            return SumSegmentTree<T>(arr.data(), arr.data() + arr.size());
          }));
}

}  // namespace rlmeta
