// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

#include <functional>

namespace py = pybind11;

namespace rlmeta {

namespace nested_utils {

py::tuple FlattenNested(const py::object& obj);

py::object MapNested(std::function<py::object(const py::object&)> func,
                     const py::object& obj);

py::object CollateNested(std::function<py::object(const py::tuple&)> func,
                         const py::tuple& src);

py::object CollateNested(std::function<py::object(const py::list&)> func,
                         const py::list& src);

py::tuple UnbatchNested(std::function<py::tuple(const py::object&)> func,
                        const py::object& obj, int64_t batch_size);

}  // namespace nested_utils

void DefineNestedUtils(py::module& m);

}  // namespace rlmeta
