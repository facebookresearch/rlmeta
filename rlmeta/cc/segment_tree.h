// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iterator>
#include <vector>

namespace rlmeta {

// SegmentTree is a tree data structure to maintain statistics of intervals.
// https://en.wikipedia.org/wiki/Segment_tree
//
// Here is the implementaion of non-recursive SegmentTree for single point
// update and interval query. The time complexities of both Update and Query are
// O(logN).
//
// One example of a SegmentTree is shown below.
//
//                          1: [0, 8)
//                       /             \
//           2: [0, 4)                      3: [4, 8)
//          /         \                    /          \
//     4: [0, 2)      5: [2, 4)      6: [4, 6)      7: [6, 8)
//     /     \        /      \        /     \        /      \
//   8: 0   9: 1   10: 2   11: 3   12: 4   13: 5   14: 6   15: 7
//
// The implementaion is adapted from the link below.
// https://codeforces.com/blog/entry/18051

template <typename T, class Operator>
class SegmentTree {
 public:
  SegmentTree(int64_t size, T identity_element)
      : size_(size),
        capacity_(Capacity(size_)),
        identity_element_(identity_element),
        values_(2 * capacity_, identity_element_) {}

  template <class InputIterator>
  SegmentTree(InputIterator first, InputIterator last, T identity_element)
      : size_(std::distance(first, last)),
        capacity_(Capacity(size_)),
        identity_element_(identity_element),
        values_(2 * capacity_, identity_element_) {
    std::copy(first, last, values_.begin() + capacity_);
    InitInternal();
  }

  int64_t size() const { return size_; }

  int64_t capacity() const { return capacity_; }

  T identity_element() const { return identity_element_; }

  T At(int64_t index) const { return values_[index + capacity_]; }

  void Reset() { std::fill(values_.begin(), values_.end(), identity_element_); }

  void Resize(int64_t size) {
    if (size < size_) {
      std::fill(values_.data() + capacity_ + size,
                values_.data() + capacity_ + size_, identity_element_);
      InitInternal();
    } else if (size > size_) {
      const int64_t cap = Capacity(size);
      if (cap > capacity_) {
        values_.resize(2 * cap, identity_element_);
        std::memcpy(values_.data() + cap, values_.data() + capacity_,
                    size_ * sizeof(T));
        capacity_ = cap;
        InitInternal();
      }
    }
    size_ = size;
  }

  void ShrinkToFit() {
    const int64_t cap = Capacity(size_);
    if (cap < capacity_) {
      std::memcpy(values_.data() + cap, values_.data() + capacity_,
                  size_ * sizeof(T));
      capacity_ = cap;
      InitInternal();
      values_.resize(2 * capacity_);
    }
  }

  template <class InputIterator>
  void Assign(InputIterator first, InputIterator last) {
    size_ = std::distance(first, last);
    capacity_ = Capacity(size_);
    values_.assign(2 * capacity_, identity_element_);
    std::copy(first, last, values_.begin() + capacity_);
  }

  // Update the item at index to value.
  // Time complexity: O(logN).
  void Update(int64_t index, T value) {
    index += capacity_;
    for (values_[index] = value; index > 1; index >>= 1) {
      values_[index >> 1] = op_(values_[index], values_[index ^ 1]);
    }
  }

  // Reduce the range of [l, r) by Operator.
  // Time complexity: O(logN)
  T Query(int64_t l, int64_t r) const {
    assert(l < r);
    if (l <= 0 && r >= size_) {
      return values_[1];
    }
    T ret = identity_element_;
    l += capacity_;
    r += capacity_;
    while (l < r) {
      if (l & 1) {
        ret = op_(ret, values_[l++]);
      }
      if (r & 1) {
        ret = op_(ret, values_[--r]);
      }
      l >>= 1;
      r >>= 1;
    }
    return ret;
  }

 protected:
  int64_t Capacity(int64_t size) const {
    int64_t capacity = 1;
    for (; capacity < size; capacity <<= 1)
      ;
    return capacity;
  }

  void InitInternal() {
    for (int64_t i = capacity_ - 1; i > 0; --i) {
      values_[i] = op_(values_[i << 1], values_[(i << 1) | 1]);
    }
  }

  const Operator op_{};
  int64_t size_;
  int64_t capacity_;
  const T identity_element_;
  std::vector<T> values_;
};

template <typename T>
class SumSegmentTree final : public SegmentTree<T, std::plus<T>> {
 public:
  explicit SumSegmentTree(int64_t size)
      : SegmentTree<T, std::plus<T>>(size, T(0)) {}

  template <class InputIterator>
  SumSegmentTree(InputIterator first, InputIterator last)
      : SegmentTree<T, std::plus<T>>(first, last, T(0)) {}

  // Get the 1st index where the scan (prefix sum) is not less than value.
  // Time complexity: O(logN)
  int64_t ScanLowerBound(T value) const {
    if (value > this->values_[1]) {
      return this->size_;
    }
    int64_t index = 1;
    T current_value = value;
    while (index < this->capacity_) {
      index <<= 1;
      const T lvalue = this->values_[index];
      if (current_value > lvalue) {
        current_value -= lvalue;
        index |= 1;
      }
    }
    return index ^ this->capacity_;
  }
};

}  // namespace rlmeta
