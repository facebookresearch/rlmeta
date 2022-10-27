// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace rlmeta {

void DefineOps(py::module& m);

}  // namespace rlmeta
