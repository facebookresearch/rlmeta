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
                 std::pair<torch::Tensor, std::vector<torch::Tensor>>& result) {
  assert(x.dim() == 1);
  const torch::Tensor x_contiguous = x.contiguous();
  const int64_t n = x_contiguous.numel();
  const T* x_data = x_contiguous.data_ptr<T>();
  std::vector<std::pair<T, int64_t>> data;
  data.reserve(n);
  for (int64_t i = 0; i < n; ++i) {
    data.emplace_back(x_data[i], i);
  }
  std::sort(data.begin(), data.end());
  std::vector<T> values;
  std::vector<torch::Tensor> groups;
  std::vector<int64_t> indices;
  if (n > 0) {
    T key = data[0].first;
    indices.push_back(data[0].second);
    for (int i = 1; i < n; ++i) {
      auto [k, v] = data[i];
      if (k != key) {
        values.push_back(key);
        groups.emplace_back(utils::AsTorchTensor(std::move(indices)));
        key = k;
      }
      indices.push_back(v);
    }
    if (!indices.empty()) {
      values.push_back(key);
      groups.emplace_back(utils::AsTorchTensor(std::move(indices)));
    }
  }
  result.first = utils::AsTorchTensor(std::move(values));
  result.second = std::move(groups);
}

}  // namespace

std::pair<torch::Tensor, std::vector<torch::Tensor>> GroupBy(
    const torch::Tensor& x) {
  std::pair<torch::Tensor, std::vector<torch::Tensor>> ret;
  AT_DISPATCH_ALL_TYPES(x.scalar_type(), "GroupBy",
                        [&]() { GroupByImpl<scalar_t>(x, ret); });
  return ret;
}

void DefineGroupByOp(py::module& m) {
  m.def("groupby", [](const torch::Tensor& x) {
    auto [values, groups] = GroupBy(x);
    const int64_t n = groups.size();
    py::tuple py_groups(n);
    for (int i = 0; i < n; ++i) {
      py_groups[i] = std::move(groups[i]);
    }
    return py::make_tuple(std::move(values), std::move(py_groups));
  });
}

}  // namespace ops
}  // namespace rlmeta
