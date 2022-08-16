// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>

namespace rlmeta {
namespace rpc {

template <typename T>
class QueueImpl {
 public:
  QueueImpl() = default;
  explicit QueueImpl(int64_t capacity) : capacity_(capacity){};

  int64_t Size() const {
    std::scoped_lock lk(mu_);
    return data_.size();
  }

  bool Empty() const {
    std::scoped_lock lk(mu_);
    return data_.empty();
  }

  bool Full() const {
    std::scoped_lock lk(mu_);
    return data_.size() == capacity_;
  }

  bool Put(const T& o) {
    {
      std::unique_lock lk(mu_);
      can_put_.wait(lk, [this]() {
        return !is_alive_ || static_cast<int64_t>(data_.size()) < capacity_;
      });
      if (!is_alive_) {
        return false;
      }
      data_.push_back(o);
    }
    can_get_.notify_one();
    return true;
  }

  bool Put(T&& o) {
    {
      std::unique_lock lk(mu_);
      can_put_.wait(lk, [this]() {
        return !is_alive_ || static_cast<int64_t>(data_.size()) < capacity_;
      });
      if (!is_alive_) {
        return false;
      }
      data_.push_back(std::move(o));
    }
    can_get_.notify_one();
    return true;
  }

  std::optional<T> Get() {
    std::optional<T> ret = [this]() -> std::optional<T> {
      std::unique_lock lk(mu_);
      can_get_.wait(lk, [this]() { return !is_alive_ || !data_.empty(); });
      if (!is_alive_) {
        return std::nullopt;
      }
      T ret = std::move(data_.front());
      data_.pop_front();
      return std::make_optional<T>(ret);
    }();
    if (ret.has_value()) {
      can_put_.notify_one();
    }
    return ret;
  }

  void Shutdown() {
    {
      std::scoped_lock lk(mu_);
      if (!is_alive_) {
        return;
      }
      is_alive_ = false;
      data_.clear();
    }
    can_put_.notify_all();
    can_get_.notify_all();
  }

 protected:
  const int64_t capacity_ = 1024;
  std::deque<T> data_;
  bool is_alive_ = true;

  std::mutex mu_;
  std::condition_variable can_put_;
  std::condition_variable can_get_;
};

}  // namespace rpc
}  // namespace rlmeta
