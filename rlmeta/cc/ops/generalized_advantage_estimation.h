// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/torch.h>

namespace py = pybind11;

namespace rlmeta {
namespace ops {

// Compute the GAE for a single episode.
// If terminated == false, len(value) == len(reward) + 1 to include the value
// for the last trucated step.
torch::Tensor GeneralizedAdvantageEstimation(const torch::Tensor& reward,
                                             const torch::Tensor& value,
                                             double gamma, double lambda,
                                             bool terminated);
torch::Tensor GeneralizedAdvantageEstimation(const torch::Tensor& reward,
                                             const torch::Tensor& value,
                                             const torch::Tensor& gamma,
                                             const torch::Tensor& lambda,
                                             bool terminated);

void DefineGeneralizedAdvantageEstimationOp(py::module& m);

}  // namespace ops
}  // namespace rlmeta
