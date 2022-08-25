// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "rlmeta/cc/utils/numpy_utils.h"
#include "rlmeta/cc/utils/torch_utils.h"

namespace rlmeta {

class TimestampManager {
 public:
  TimestampManager() = default;
  explicit TimestampManager(int64_t size) : size_(size) { Reset(); }

  int64_t size() const { return size_; }

  int64_t current_timestamp() const { return current_timestamp_; }

  void Reset() {
    timestamps_.assign(size_, -1);
    current_timestamp_ = 0;
  }

  void Reset(int64_t size) {
    size_ = size;
    timestamps_.assign(size_, -1);
    current_timestamp_ = 0;
  }

  int64_t At(int64_t index) const { return timestamps_[index]; }

  py::array_t<int64_t> At(const py::array_t<int64_t>& index) const {
    py::array_t<int64_t> timestamp =
        utils::NumpyEmptyLike<int64_t, int64_t>(index);
    BatchAtImpl(index.size(), index.data(), timestamp.mutable_data());
    return timestamp;
  }

  torch::Tensor At(const torch::Tensor& index) const {
    assert(index.dtype() == torch::kInt64);
    const torch::Tensor index_contiguous = index.contiguous();
    const int64_t n = index_contiguous.numel();
    torch::Tensor timestamp = torch::empty_like(index_contiguous);
    BatchAtImpl(n, index_contiguous.data_ptr<int64_t>(),
                timestamp.data_ptr<int64_t>());
    return timestamp;
  }

  py::array_t<int64_t> Numpy() const {
    py::array_t<int64_t> arr(size_);
    std::memcpy(arr.mutable_data(), timestamps_.data(),
                size_ * sizeof(int64_t));
    return arr;
  }

  bool IsAvailable(int64_t index, int64_t timestamp) const {
    return timestamps_[index] == timestamp;
  }

  py::array_t<bool> IsAvailable(const py::array_t<int64_t>& index,
                                const py::array_t<int64_t>& timestamp) const {
    py::array_t<bool> mask = utils::NumpyEmptyLike<int64_t, bool>(index);
    BatchIsAvailableImpl(index.size(), index.data(), timestamp.data(),
                         mask.mutable_data());
    return mask;
  }

  torch::Tensor IsAvailable(const torch::Tensor& index,
                            const torch::Tensor& timestamp) const {
    assert(index.dtype() == torch::kInt64);
    assert(timestamp.dtype() == torch::kInt64);
    const torch::Tensor index_contiguous = index.contiguous();
    const torch::Tensor timestamp_contiguous = timestamp.contiguous();
    const int64_t n = index_contiguous.numel();
    torch::Tensor mask =
        torch::empty_like(index_contiguous, utils::TorchDataType<bool>::value);
    BatchIsAvailableImpl(n, index_contiguous.data_ptr<int64_t>(),
                         timestamp_contiguous.data_ptr<int64_t>(),
                         mask.data_ptr<bool>());
    return mask;
  }

  void Update(int64_t index) { timestamps_[index] = current_timestamp_++; }

  void Update(const py::array_t<int64_t>& index) {
    BatchUpdateImpl(index.size(), index.data());
  }

  void Update(const py::array_t<int64_t>& index,
              const py::array_t<bool>& mask) {
    BatchUpdateImpl(index.size(), index.data(), mask.data());
  }

  void Update(const torch::Tensor& index) {
    assert(index.dtype() == torch::kInt64);
    const torch::Tensor index_contiguous = index.contiguous();
    BatchUpdateImpl(index_contiguous.numel(),
                    index_contiguous.data_ptr<int64_t>());
  }

  void Update(const torch::Tensor& index, const torch::Tensor& mask) {
    assert(index.dtype() == torch::kInt64);
    assert(mask.dtype() == torch::kBool);
    const torch::Tensor index_contiguous = index.contiguous();
    const torch::Tensor mask_contiguous = mask.contiguous();
    BatchUpdateImpl(index_contiguous.numel(),
                    index_contiguous.data_ptr<int64_t>(),
                    mask_contiguous.data_ptr<bool>());
  }

  void LoadData(const py::array_t<int64_t>& data, int64_t current_timestamp) {
    size_ = data.size();
    timestamps_.resize(size_);
    std::memcpy(timestamps_.data(), data.data(), size_ * sizeof(int64_t));
    current_timestamp_ = current_timestamp;
  }

 protected:
  void BatchAtImpl(int64_t n, const int64_t* index, int64_t* timestamp) const;

  void BatchIsAvailableImpl(int64_t n, const int64_t* index,
                            const int64_t* timestamp, bool* mask) const;

  void BatchUpdateImpl(int64_t n, const int64_t* index);
  void BatchUpdateImpl(int64_t n, const int64_t* index, const bool* mask);

  int64_t size_;
  std::vector<int64_t> timestamps_;
  int64_t current_timestamp_;
};

void DefineTimestampManager(py::module& m);

}  // namespace rlmeta
