// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "rlmeta/rpc/cc/blocking_counter.h"

#include <cassert>

namespace rlmeta {
namespace rpc {

bool BlockingCounter::DecrementCount() {
  bool ret = false;
  {
    std::unique_lock<std::mutex> lk(mu_);
    assert(count_ >= 1);
    --count_;
    ret = (count_ == 0);
  }
  cv_.notify_all();
  return ret;
}

void BlockingCounter::Wait() {
  std::unique_lock<std::mutex> lk(mu_);
  assert(num_waiting_ == 0);
  ++num_waiting_;
  cv_.wait(lk, [this]() { return count_ == 0; });
}

}  // namespace rpc
}  // namespace rlmeta
