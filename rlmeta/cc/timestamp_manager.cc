// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "rlmeta/cc/timestamp_manager.h"

#include <memory>

namespace rlmeta {

void TimestampManager::BatchAtImpl(int64_t n, const int64_t* index,
                                   int64_t* timestamp) const {
  for (int64_t i = 0; i < n; ++i) {
    timestamp[i] = timestamps_[index[i]];
  }
}

void TimestampManager::BatchIsAvailableImpl(int64_t n, const int64_t* index,
                                            const int64_t* timestamp,
                                            bool* mask) const {
  for (int64_t i = 0; i < n; ++i) {
    mask[i] = (timestamp[i] == timestamps_[index[i]]);
  }
}

void TimestampManager::BatchUpdateImpl(int64_t n, const int64_t* index) {
  for (int64_t i = 0; i < n; ++i) {
    timestamps_[index[i]] = current_timestamp_;
  }
  ++current_timestamp_;
}

void TimestampManager::BatchUpdateImpl(int64_t n, const int64_t* index,
                                       const bool* mask) {
  for (int64_t i = 0; i < n; ++i) {
    if (mask[i]) {
      timestamps_[index[i]] = current_timestamp_;
    }
  }
  ++current_timestamp_;
}

void DefineTimestampManager(py::module& m) {
  py::class_<TimestampManager, std::shared_ptr<TimestampManager>>(
      m, "TimestampManager")
      .def(py::init<int64_t>())
      .def_property_readonly("size", &TimestampManager::size)
      .def_property_readonly("current_timestamp",
                             &TimestampManager::current_timestamp)
      .def("__len__", &TimestampManager::size)
      .def("__getitem__",
           py::overload_cast<int64_t>(&TimestampManager::At, py::const_))
      .def("__getitem__", py::overload_cast<const py::array_t<int64_t>&>(
                              &TimestampManager::At, py::const_))
      .def("__getitem__", py::overload_cast<const torch::Tensor&>(
                              &TimestampManager::At, py::const_))
      .def("at", py::overload_cast<int64_t>(&TimestampManager::At, py::const_))
      .def("at", py::overload_cast<const py::array_t<int64_t>&>(
                     &TimestampManager::At, py::const_))
      .def("at", py::overload_cast<const torch::Tensor&>(&TimestampManager::At,
                                                         py::const_))
      .def("is_available", py::overload_cast<int64_t, int64_t>(
                               &TimestampManager::IsAvailable, py::const_))
      .def("is_available", py::overload_cast<const py::array_t<int64_t>&,
                                             const py::array_t<int64_t>&>(
                               &TimestampManager::IsAvailable, py::const_))
      .def("is_available",
           py::overload_cast<const torch::Tensor&, const torch::Tensor&>(
               &TimestampManager::IsAvailable, py::const_))
      .def("update", py::overload_cast<int64_t>(&TimestampManager::Update))
      .def("update", py::overload_cast<const py::array_t<int64_t>&>(
                         &TimestampManager::Update))
      .def("update",
           py::overload_cast<const torch::Tensor&>(&TimestampManager::Update))
      .def("update", py::overload_cast<const py::array_t<int64_t>&,
                                       const py::array_t<bool>&>(
                         &TimestampManager::Update))
      .def("update",
           py::overload_cast<const torch::Tensor&, const torch::Tensor&>(
               &TimestampManager::Update))
      .def(py::pickle(
          [](const TimestampManager& m) {
            return py::make_tuple(m.Numpy(), m.current_timestamp());
          },
          [](const py::tuple& t) {
            assert(t.size() == 2);
            const py::array_t<int64_t> data = t[0].cast<py::array_t<int64_t>>();
            const int64_t current_timestamp = t[1].cast<int64_t>();
            TimestampManager timestamps;
            timestamps.LoadData(data, current_timestamp);
            return timestamps;
          }));
}

}  // namespace rlmeta
