// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "rlmeta/cc/ops/groupby.h"

#include <algorithm>
#include <cassert>

#include "rlmeta/cc/utils/torch_utils.h"

namespace rlmeta {
namespace ops {

namespace {

template <typename T>
void GroupByImpl(const torch::Tensor& x,
                 std::vector<std::pair<torch::Tensor, torch::Tensor>>& result) {
  assert(x.dim() == 1);
  const torch::Tensor x_contiguous = x.contiguous();
  const int64_t n = x_contiguous.numel();
  const T* x_data = x_contiguous.data_ptr<T>();
  if (n == 0) {
    return;
  }
  std::vector<std::pair<T, int64_t>> data;
  data.reserve(n);
  for (int64_t i = 0; i < n; ++i) {
    data.emplace_back(x_data[i], i);
  }
  std::sort(data.begin(), data.end());
  result.clear();
  result.reserve(n);
  std::vector<int64_t> indices;
  T key = data[0].first;
  indices.push_back(data[0].second);
  for (int i = 1; i < n; ++i) {
    auto [k, v] = data[i];
    if (k != key) {
      result.emplace_back(torch::tensor(key, utils::TorchDataType<T>::value),
                          utils::AsTorchTensor(std::move(indices)));
      key = k;
    }
    indices.push_back(v);
  }
  if (!indices.empty()) {
    result.emplace_back(torch::tensor(key, utils::TorchDataType<T>::value),
                        utils::AsTorchTensor(std::move(indices)));
  }
}

}  // namespace

std::vector<std::pair<torch::Tensor, torch::Tensor>> GroupBy(
    const torch::Tensor& x) {
  std::vector<std::pair<torch::Tensor, torch::Tensor>> ret;
  AT_DISPATCH_ALL_TYPES_AND(torch::kBool, x.scalar_type(), "GroupBy",
                            [&]() { GroupByImpl<scalar_t>(x, ret); });
  return ret;
}

void DefineGroupByOp(py::module& m) {
  m.def("groupby", [](const torch::Tensor& x) {
    std::vector<std::pair<torch::Tensor, torch::Tensor>> vec = GroupBy(x);
    const int64_t n = vec.size();
    py::tuple ret(n);
    for (int i = 0; i < n; ++i) {
      ret[i] =
          py::make_tuple(std::move(vec[i].first), std::move(vec[i].second));
    }
    return ret;
  });
}

}  // namespace ops
}  // namespace rlmeta
