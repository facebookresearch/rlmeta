// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "rlmeta/cc/samplers/sampler.h"

#include <pybind11/stl.h>

#include <memory>

namespace rlmeta {

void DefineSampler(py::module& m) {
  py::class_<Sampler, PySampler, std::shared_ptr<Sampler>>(m, "Sampler")
      .def_property_readonly("size", &Sampler::Size)
      .def("__len__", &Sampler::Size)
      .def("reset", py::overload_cast<>(&Sampler::Reset))
      .def("reset", py::overload_cast<int64_t>(&Sampler::Reset))
      .def("insert", py::overload_cast<int64_t, double>(&Sampler::Insert))
      .def("insert", py::overload_cast<const py::array_t<int64_t>&, double>(
                         &Sampler::Insert))
      .def("insert",
           py::overload_cast<const py::array_t<int64_t>&,
                             const py::array_t<double>&>(&Sampler::Insert))
      .def("insert",
           py::overload_cast<const torch::Tensor&, double>(&Sampler::Insert))
      .def("insert",
           py::overload_cast<const torch::Tensor&, const torch::Tensor&>(
               &Sampler::Insert))
      .def("update", py::overload_cast<int64_t, double>(&Sampler::Update))
      .def("update", py::overload_cast<const py::array_t<int64_t>&, double>(
                         &Sampler::Update))
      .def("update",
           py::overload_cast<const py::array_t<int64_t>&,
                             const py::array_t<double>&>(&Sampler::Update))
      .def("update",
           py::overload_cast<const torch::Tensor&, double>(&Sampler::Update))
      .def("update",
           py::overload_cast<const torch::Tensor&, const torch::Tensor&>(
               &Sampler::Update))
      .def("delete", py::overload_cast<int64_t>(&Sampler::Delete))
      .def("delete",
           py::overload_cast<const py::array_t<int64_t>&>(&Sampler::Delete))
      .def("delete", py::overload_cast<const torch::Tensor&>(&Sampler::Delete))
      .def("sample", &Sampler::Sample, py::arg("num_samples"),
           py::arg("replacement") = false);
}

}  // namespace rlmeta
