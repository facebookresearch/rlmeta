// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "rlmeta/cc/storage/schema.h"

#include "rlmeta/cc/nested_utils/nested_utils.h"
#include "rlmeta/cc/utils/torch_utils.h"

namespace rlmeta {

void Schema::Reset() {
  if (meta_.has_value()) {
    meta_.reset();
    return;
  }
  if (vec_.has_value()) {
    for (auto& sub : *vec_) {
      sub.Reset();
    }
    vec_->clear();
    vec_.reset();
    return;
  }
  if (map_.has_value()) {
    for (auto& [key, sub] : *map_) {
      sub.Reset();
    }
    map_->clear();
    map_.reset();
  }
}

bool Schema::FromPythonImpl(const py::object& obj, bool packed_input,
                            int64_t& index) {
  if (utils::IsTorchTensor(obj)) {
    const torch::Tensor src = utils::PyObjectToTorchTensor(obj);
    size_ = 1;
    meta_.emplace();
    meta_->index = index++;
    meta_->dtype = static_cast<int64_t>(src.scalar_type());
    meta_->shape.assign(
        src.sizes().cbegin() + static_cast<int64_t>(packed_input),
        src.sizes().cend());
    return true;
  }

  if (py::isinstance<py::tuple>(obj)) {
    const py::tuple src = py::reinterpret_borrow<py::tuple>(obj);
    size_ = 0;
    vec_.emplace();
    vec_->reserve(src.size());
    for (const auto x : src) {
      Schema& cur = vec_->emplace_back();
      if (!cur.FromPythonImpl(py::reinterpret_borrow<py::object>(x),
                              packed_input, index)) {
        size_ = 0;
        vec_.reset();
        return false;
      }
      size_ += cur.size();
    }
    return true;
  }

  if (py::isinstance<py::list>(obj)) {
    const py::list src = py::reinterpret_borrow<py::list>(obj);
    size_ = 0;
    vec_.emplace();
    vec_->reserve(src.size());
    for (const auto x : src) {
      Schema& cur = vec_->emplace_back();
      if (!cur.FromPythonImpl(py::reinterpret_borrow<py::object>(x),
                              packed_input, index)) {
        size_ = 0;
        vec_.reset();
        return false;
      }
      size_ += cur.size();
    }
    return true;
  }

  if (py::isinstance<py::dict>(obj)) {
    const py::dict src = py::reinterpret_borrow<py::dict>(obj);
    const std::vector<std::string> keys = nested_utils::SortedKeys(src);
    size_ = 0;
    map_.emplace();
    map_->reserve(keys.size());
    for (const std::string& k : keys) {
      auto& cur = map_->emplace_back(k, Schema());
      if (!cur.second.FromPythonImpl(
              py::reinterpret_borrow<py::object>(src[py::str(k)]), packed_input,
              index)) {
        size_ = 0;
        map_.reset();
        return false;
      }
      size_ += cur.second.size();
    }
    return true;
  }

  size_ = 0;
  return false;
}

}  // namespace rlmeta
