// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <pybind11/pybind11.h>

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace rlmeta {

class Schema {
 public:
  struct MetaData {
    int64_t index;
    int64_t dtype;
    std::vector<int64_t> shape;
  };

  Schema() = default;

  int64_t size() const { return size_; }

  void Reset();

  const std::optional<MetaData>& meta() const { return meta_; }

  const std::optional<std::vector<Schema>>& vec() const { return vec_; }

  const std::optional<std::vector<std::pair<std::string, Schema>>>& map()
      const {
    return map_;
  }

  bool FromPython(const py::object& obj, bool packed_input = false) {
    int64_t index = 0;
    return FromPythonImpl(obj, packed_input, index);
  }

 protected:
  bool FromPythonImpl(const py::object& obj, bool packed_input, int64_t& index);

  int64_t size_ = 0;
  std::optional<MetaData> meta_ = std::nullopt;
  std::optional<std::vector<Schema>> vec_ = std::nullopt;
  std::optional<std::vector<std::pair<std::string, Schema>>> map_ =
      std::nullopt;
};

}  // namespace rlmeta
