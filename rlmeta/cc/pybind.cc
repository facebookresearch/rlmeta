// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <pybind11/pybind11.h>

#include "rlmeta/cc/circular_buffer.h"
#include "rlmeta/cc/nested_utils.h"
#include "rlmeta/cc/segment_tree.h"
#include "rlmeta/cc/timestamp_manager.h"

namespace py = pybind11;

namespace {

PYBIND11_MODULE(_rlmeta_extension, m) {
  rlmeta::DefineSumSegmentTree<float>("Fp32", m);
  rlmeta::DefineSumSegmentTree<double>("Fp64", m);
  rlmeta::DefineMinSegmentTree<float>("Fp32", m);
  rlmeta::DefineMinSegmentTree<double>("Fp64", m);

  rlmeta::DefineCircularBuffer(m);
  rlmeta::DefineNestedUtils(m);
  rlmeta::DefineTimestampManager(m);
}

}  // namespace
