// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <pybind11/pybind11.h>

#include "rlmeta/cc/circular_buffer.h"
#include "rlmeta/cc/nested_utils.h"
#include "rlmeta/cc/segment_tree.h"
#include "rlmeta/cc/timestamp_manager.h"
#include "rlmeta/rpc/cc/computation_queue.h"
#include "rlmeta/rpc/cc/server.h"
#include "rlmeta/rpc/cc/task.h"

namespace py = pybind11;

namespace {

PYBIND11_MODULE(_rlmeta_extension, m) {
  rlmeta::DefineSumSegmentTree<float>("Fp32", m);
  rlmeta::DefineSumSegmentTree<double>("Fp64", m);
  rlmeta::DefineMinSegmentTree<float>("Fp32", m);
  rlmeta::DefineMinSegmentTree<double>("Fp64", m);

  rlmeta::DefineCircularBuffer(m);
  rlmeta::DefineTimestampManager(m);

  py::module nested_utils = m.def_submodule(
      "nested_utils", "A submodule of \"_rlmeta_extension\" for nested_utils");
  rlmeta::DefineNestedUtils(nested_utils);

  py::module rpc =
      m.def_submodule("rpc", "A submodule of \"_rlmeta_extension\" for RPC");
  rlmeta::rpc::DefineTaskBase(rpc);
  rlmeta::rpc::DefineTask(rpc);
  rlmeta::rpc::DefineBatchedTask(rpc);
  rlmeta::rpc::DefineComputationQueue(rpc);
  rlmeta::rpc::DefineBatchedComputationQueue(rpc);
  rlmeta::rpc::DefineServer(rpc);
}

}  // namespace
