// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <optional>

namespace py = pybind11;

namespace rlmeta {
namespace ops {

// Compute the GAE for a single episode.
// last_v is used for truncated trajectories.
torch::Tensor GeneralizedAdvantageEstimation(
    const torch::Tensor& reward, const torch::Tensor& value, double gamma,
    double lambda, const std::optional<torch::Tensor>& last_v = std::nullopt);

torch::Tensor GeneralizedAdvantageEstimation(
    const torch::Tensor& reward, const torch::Tensor& value,
    const torch::Tensor& gamma, const torch::Tensor& lambda,
    const std::optional<torch::Tensor>& last_v = std::nullopt);

void DefineGeneralizedAdvantageEstimationOp(py::module& m);

}  // namespace ops
}  // namespace rlmeta
