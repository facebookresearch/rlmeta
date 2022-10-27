// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "rlmeta/cc/ops/ops.h"

#include "rlmeta/cc/ops/groupby.h"

namespace rlmeta {

void DefineOps(py::module& m) {
  py::module sub =
      m.def_submodule("ops", "ops submodule of \"_rlmeta_extension\"");

  ops::DefineGroupByOp(sub);
}

}  // namespace rlmeta
