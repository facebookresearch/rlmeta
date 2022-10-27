// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <pybind11/pybind11.h>

#include "rlmeta/cc/nested_utils/nested_utils.h"
#include "rlmeta/cc/ops/ops.h"
#include "rlmeta/cc/samplers/prioritized_sampler.h"
#include "rlmeta/cc/samplers/sampler.h"
#include "rlmeta/cc/samplers/uniform_sampler.h"
#include "rlmeta/cc/storage/circular_buffer.h"
#include "rlmeta/cc/storage/tensor_circular_buffer.h"
#include "rlmeta/cc/utils/segment_tree_pybind.h"

namespace py = pybind11;

namespace {

PYBIND11_MODULE(_rlmeta_extension, m) {
  rlmeta::DefineSumSegmentTree<float>("Fp32", m);
  rlmeta::DefineSumSegmentTree<double>("Fp64", m);

  rlmeta::DefineCircularBuffer(m);
  rlmeta::DefineTensorCircularBuffer(m);

  rlmeta::DefineNestedUtils(m);

  rlmeta::DefineSampler(m);
  rlmeta::DefineUniformSampler(m);
  rlmeta::DefinePrioritizedSampler(m);

  rlmeta::DefineOps(m);
}

}  // namespace
