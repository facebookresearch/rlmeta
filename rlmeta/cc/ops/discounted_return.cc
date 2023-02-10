// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "rlmeta/cc/ops/discounted_return.h"

#include <cassert>

#include "rlmeta/cc/utils/torch_utils.h"

namespace rlmeta {
namespace ops {

namespace {

template <typename T>
void DiscountedReturnImpl(const torch::Tensor& reward, T gamma,
                          torch::Tensor& discounted_return) {
  assert(reward.dim() == 1 || (reward.dim() == 2 && reward.size(1) == 1));
  const torch::Tensor reward_contiguous = reward.contiguous();
  const int64_t n = reward_contiguous.numel();
  const T* r_data = reward_contiguous.data_ptr<T>();
  T* g_data = discounted_return.data_ptr<T>();
  T g = 0;
  for (int i = n - 1; i >= 0; --i) {
    g = r_data[i] + gamma * g;
    g_data[i] = g;
  }
}

template <typename T>
void DiscountedReturnImpl(const torch::Tensor& reward,
                          const torch::Tensor& gamma,
                          torch::Tensor& discounted_return) {
  assert(reward.dim() == 1 || (reward.dim() == 2 && reward.size(1) == 1));
  assert(gamma.numel() == 1 || gamma.sizes() == reward.sizes());

  if (gamma.numel() == 1) {
    DiscountedReturnImpl<T>(reward, gamma.item<T>(), discounted_return);
    return;
  }

  const torch::Tensor reward_contiguous = reward.contiguous();
  const torch::Tensor gamma_coutiguous = gamma.contiguous();
  const int64_t n = reward_contiguous.numel();
  const T* r_data = reward_contiguous.data_ptr<T>();
  const T* gamma_data = gamma_coutiguous.data_ptr<T>();
  T* g_data = discounted_return.data_ptr<T>();
  T g = 0;
  for (int i = n - 1; i >= 0; --i) {
    g = r_data[i] + gamma_data[i] * g;
    g_data[i] = g;
  }
}

}  // namespace

torch::Tensor DiscountedReturn(const torch::Tensor& reward, double gamma) {
  torch::Tensor discounted_return = torch::empty_like(reward);
  AT_DISPATCH_FLOATING_TYPES(reward.scalar_type(), "DiscountedReturn", [&]() {
    DiscountedReturnImpl<scalar_t>(reward, static_cast<scalar_t>(gamma),
                                   discounted_return);
  });
  return discounted_return;
}

torch::Tensor DiscountedReturn(const torch::Tensor& reward,
                               const torch::Tensor& gamma) {
  torch::Tensor discounted_return = torch::empty_like(reward);
  AT_DISPATCH_FLOATING_TYPES(reward.scalar_type(), "DiscountedReturn", [&]() {
    DiscountedReturnImpl<scalar_t>(reward, gamma, discounted_return);
  });
  return discounted_return;
}

void DefineDiscountedReturnOp(py::module& m) {
  m.def("discounted_return",
        py::overload_cast<const torch::Tensor&, double>(&DiscountedReturn))
      .def("discounted_return",
           py::overload_cast<const torch::Tensor&, const torch::Tensor&>(
               &DiscountedReturn));
}

}  // namespace ops
}  // namespace rlmeta
