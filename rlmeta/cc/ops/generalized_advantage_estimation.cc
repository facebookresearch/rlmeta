// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "rlmeta/cc/ops/generalized_advantage_estimation.h"

#include <cassert>

#include "rlmeta/cc/utils/torch_utils.h"

namespace rlmeta {
namespace ops {

namespace {

template <typename T>
T ValueAt(const T* v, int64_t n, int64_t k) {
  return n == 1 ? v[0] : v[k];
}

template <typename T>
void GeneralizedAdvantageEstimationImpl(const torch::Tensor& reward,
                                        const torch::Tensor& value, T gamma,
                                        T lambda, bool terminated,
                                        torch::Tensor& gae) {
  assert(reward.dim() == 1 || (reward.dim() == 2 && reward.size(1) == 1));
  assert(value.dim() == 1 || (value.dim() == 2 && value.size(1) == 1));
  const int64_t n = reward.size(0);
  if (terminated) {
    assert(value.size(0) == n || value.size(0) == n + 1);
  } else {
    assert(value.size(0) == n + 1);
  }

  const torch::Tensor reward_contiguous = reward.contiguous();
  const torch::Tensor value_contiguous = value.contiguous();
  const T* r_data = reward_contiguous.data_ptr<T>();
  const T* v_data = value_contiguous.data_ptr<T>();
  T* gae_data = gae.data_ptr<T>();
  T v = terminated ? 0 : v_data[n];
  T adv = 0;
  for (int64_t i = n - 1; i >= 0; --i) {
    const T delta = r_data[i] + gamma * v - v_data[i];
    v = v_data[i];
    adv = delta + gamma * lambda * adv;
    gae_data[i] = adv;
  }
}

template <typename T>
void GeneralizedAdvantageEstimationImpl(const torch::Tensor& reward,
                                        const torch::Tensor& value,
                                        const torch::Tensor& gamma,
                                        const torch::Tensor& lambda,
                                        bool terminated, torch::Tensor& gae) {
  assert(reward.dim() == 1 || (reward.dim() == 2 && reward.size(1) == 1));
  assert(value.dim() == 1 || (value.dim() == 2 && value.size(1) == 1));
  const int64_t n = reward.size(0);
  if (terminated) {
    assert(value.size(0) == n || value.size(0) == n + 1);
  } else {
    assert(value.size(0) == n + 1);
  }
  assert(gamma.numel() == 1 || gamma.sizes() == reward.sizes());
  assert(lambda.numel() == 1 || lambda.sizes() == reward.sizes());

  const torch::Tensor reward_contiguous = reward.contiguous();
  const torch::Tensor value_contiguous = value.contiguous();
  const torch::Tensor gamma_contiguous = gamma.contiguous();
  const torch::Tensor lambda_contiguous = lambda.contiguous();
  const T* r_data = reward_contiguous.data_ptr<T>();
  const T* v_data = value_contiguous.data_ptr<T>();
  const T* gamma_data = gamma_contiguous.data_ptr<T>();
  const T* lambda_data = lambda_contiguous.data_ptr<T>();
  const int64_t gamma_n = gamma_contiguous.numel();
  const int64_t lambda_n = lambda_contiguous.numel();
  T* gae_data = gae.data_ptr<T>();
  T v = terminated ? 0 : v_data[n];
  T adv = 0;
  for (int64_t i = n - 1; i >= 0; --i) {
    const T gamma_i = ValueAt(gamma_data, gamma_n, i);
    const T lambda_i = ValueAt(lambda_data, lambda_n, i);
    const T delta = r_data[i] + gamma_i * v - v_data[i];
    v = v_data[i];
    adv = delta + gamma_i * lambda_i * adv;
    gae_data[i] = adv;
  }
}

}  // namespace

torch::Tensor GeneralizedAdvantageEstimation(const torch::Tensor& reward,
                                             const torch::Tensor& value,
                                             double gamma, double lambda,
                                             bool terminated) {
  torch::Tensor gae = torch::empty_like(reward);
  AT_DISPATCH_FLOATING_TYPES(
      reward.scalar_type(), "GeneralizedAdvantageEstimation", [&]() {
        GeneralizedAdvantageEstimationImpl<scalar_t>(
            reward, value, static_cast<scalar_t>(gamma),
            static_cast<scalar_t>(lambda), terminated, gae);
      });
  return gae;
}

torch::Tensor GeneralizedAdvantageEstimation(const torch::Tensor& reward,
                                             const torch::Tensor& value,
                                             const torch::Tensor& gamma,
                                             const torch::Tensor& lambda,
                                             bool terminated) {
  torch::Tensor gae = torch::empty_like(reward);
  AT_DISPATCH_FLOATING_TYPES(
      reward.scalar_type(), "GeneralizedAdvantageEstimation", [&]() {
        GeneralizedAdvantageEstimationImpl<scalar_t>(reward, value, gamma,
                                                     lambda, terminated, gae);
      });
  return gae;
}

void DefineGeneralizedAdvantageEstimationOp(py::module& m) {
  m.def("generalized_advantage_estimation",
        py::overload_cast<const torch::Tensor&, const torch::Tensor&, double,
                          double, bool>(&GeneralizedAdvantageEstimation),
        py::arg("reward"), py::arg("value"), py::arg("gamma"),
        py::arg("gae_lambda"), py::arg("terminated"))
      .def("generalized_advantage_estimation",
           py::overload_cast<const torch::Tensor&, const torch::Tensor&,
                             const torch::Tensor&, const torch::Tensor&, bool>(
               &GeneralizedAdvantageEstimation),
           py::arg("reward"), py::arg("value"), py::arg("gamma"),
           py::arg("gae_lambda"), py::arg("terminated"));
}

}  // namespace ops
}  // namespace rlmeta
