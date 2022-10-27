// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <utility>
#include <vector>

namespace py = pybind11;

namespace rlmeta {
namespace ops {

std::pair<torch::Tensor, std::vector<torch::Tensor>> GroupBy(
    const torch::Tensor& x);

void DefineGroupByOp(py::module& m);

}  // namespace ops
}  // namespace rlmeta
