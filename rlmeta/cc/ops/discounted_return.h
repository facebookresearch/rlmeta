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

// Compute the discounted_return for a single episode.
// The last terminated or truncated step is not included.
torch::Tensor DiscountedReturn(const torch::Tensor& reward, double gamma);
torch::Tensor DiscountedReturn(const torch::Tensor& reward,
                               const torch::Tensor& gamma);

// TODO: Implement the DiscountedReturn function with terminated flag.
//
// torch::Tensor DiscountedReturn(const torch::Tensor& reward, double gamma,
//                                const torch::Tensor& terminated);
// torch::Tensor DiscountedReturn(const torch::Tensor& reward,
//                                const torch::Tensor& gamma,
//                                const torch::Tensor& terminated);

void DefineDiscountedReturnOp(py::module& m);

}  // namespace ops
}  // namespace rlmeta
