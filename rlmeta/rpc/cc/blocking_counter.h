// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>

namespace rlmeta {
namespace rpc {

// https://github.com/abseil/abseil-cpp/blob/ce42de10fbea616379826e91c7c23c16bffe6e61/absl/synchronization/blocking_counter.h

class BlockingCounter {
 public:
  explicit BlockingCounter(int64_t initial_count) : count_(initial_count) {}

  BlockingCounter(const BlockingCounter&) = delete;
  BlockingCounter& operator=(const BlockingCounter&) = delete;

  bool DecrementCount();

  void Wait();

 private:
  int64_t count_;
  int64_t num_waiting_ = 0;

  std::mutex mu_;
  std::condition_variable cv_;
};

}  // namespace rpc
}  // namespace rlmeta
