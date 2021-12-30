// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <pybind11/pybind11.h>

#include "rlmeta/cc/circular_buffer.h"
#include "rlmeta/cc/nested_utils.h"
#include "rlmeta/cc/segment_tree.h"

namespace py = pybind11;

namespace {

PYBIND11_MODULE(rlmeta_extension, m) {
  rlmeta::DefineSumSegmentTree<float>("Float", m);
  rlmeta::DefineSumSegmentTree<double>("Double", m);

  rlmeta::DefineMinSegmentTree<float>("Float", m);
  rlmeta::DefineMinSegmentTree<double>("Double", m);

  rlmeta::DefineCircularBuffer(m);

  rlmeta::DefineNestedUtils(m);
}

}  // namespace
