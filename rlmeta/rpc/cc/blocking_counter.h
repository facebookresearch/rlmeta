// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <condition_variable>
#include <cstdint>
#include <mutex>

namespace rlmeta {
namespace rpc {

// BlockingCounter implementation is copied and modified from
// the BlockingCounter class in Abseil.
// https://github.com/abseil/abseil-cpp/blob/ce42de10fbea616379826e91c7c23c16bffe6e61/absl/synchronization/blocking_counter.h
//
// Copyright 2017 The Abseil Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
