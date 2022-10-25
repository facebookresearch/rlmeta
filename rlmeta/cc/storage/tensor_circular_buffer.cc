// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "rlmeta/cc/storage/tensor_circular_buffer.h"

#include <algorithm>
#include <cassert>
#include <memory>
#include <thread>

#include "rlmeta/cc/nested_utils/nested_utils.h"
#include "rlmeta/cc/utils/torch_utils.h"

namespace rlmeta {

namespace {

void ReserveTensors(const Schema& schema, int64_t capacity,
                    std::vector<torch::Tensor>& tensors) {
  if (schema.meta().has_value()) {
    std::vector<int64_t> shape(schema.meta()->shape.size() + 1);
    shape[0] = capacity;
    std::copy(schema.meta()->shape.cbegin(), schema.meta()->shape.cend(),
              shape.begin() + 1);
    tensors[schema.meta()->index] = torch::empty(
        shape, torch::TensorOptions().dtype(
                   static_cast<at::ScalarType>(schema.meta()->dtype)));
    return;
  }
  if (schema.vec().has_value()) {
    for (const Schema& sub : *schema.vec()) {
      ReserveTensors(sub, capacity, tensors);
    }
    return;
  }
  if (schema.map().has_value()) {
    for (const auto& [key, sub] : *schema.map()) {
      ReserveTensors(sub, capacity, tensors);
    }
    return;
  }
}

void FlattenNested(const Schema& schema, const py::object& src,
                   std::vector<torch::Tensor>& dst) {
  if (utils::IsTorchTensor(src)) {
    assert(schema.meta().has_value());
    dst[schema.meta()->index] = utils::PyObjectToTorchTensor(src);
    return;
  }
  if (py::isinstance<py::tuple>(src)) {
    const py::tuple src_vec = py::reinterpret_borrow<py::tuple>(src);
    const int64_t n = src_vec.size();
    assert(schema.vec().has_value());
    assert(schema.vec()->size() == n);
    for (int64_t i = 0; i < n; ++i) {
      FlattenNested(schema.vec()->at(i), src_vec[i], dst);
    }
    return;
  }
  if (py::isinstance<py::list>(src)) {
    const py::list src_vec = py::reinterpret_borrow<py::list>(src);
    const int64_t n = src_vec.size();
    assert(schema.vec().has_value());
    assert(schema.vec()->size() == n);
    for (int64_t i = 0; i < n; ++i) {
      FlattenNested(schema.vec()->at(i), src_vec[i], dst);
    }
    return;
  }
  if (py::isinstance<py::dict>(src)) {
    const py::dict src_map = py::reinterpret_borrow<py::dict>(src);
    assert(schema.map().has_value());
    for (const auto& [key, sub] : *schema.map()) {
      FlattenNested(sub, src_map[py::str(key)], dst);
    }
    return;
  }
}

std::vector<int64_t> ComputeWorkloads(int64_t num_jobs, int64_t num_workers) {
  const int64_t n = std::min(num_jobs, num_workers);
  const int64_t m = num_jobs / n;
  const int64_t r = num_jobs % n;
  std::vector<int64_t> ret(n + 1, 0);
  for (int64_t i = 0; i < n; ++i) {
    ret[i + 1] = ret[i] + m + (i < r);
  }
  return ret;
}

py::object RecoverNestedImpl(const Schema& schema,
                             const std::vector<torch::Tensor>& src) {
  if (schema.meta().has_value()) {
    return utils::TorchTensorToPyObject(src.at(schema.meta()->index));
  }
  if (schema.vec().has_value()) {
    const int64_t n = schema.vec()->size();
    py::tuple dst(n);
    for (int64_t i = 0; i < n; ++i) {
      dst[i] = RecoverNestedImpl(schema.vec()->at(i), src);
    }
    return std::move(dst);
  }
  if (schema.map().has_value()) {
    py::dict dst;
    for (const auto& [key, sub] : *schema.map()) {
      dst[py::str(key)] = RecoverNestedImpl(sub, src);
    }
    return std::move(dst);
  }
  return py::none();
}

void LoadNested(const Schema& schema, const py::object& src,
                std::vector<torch::Tensor>& dst) {
  if (utils::IsTorchTensor(src)) {
    assert(schema.meta().has_value());
    dst[schema.meta()->index] = utils::PyObjectToTorchTensor(src);
    return;
  }
  if (py::isinstance<py::tuple>(src)) {
    const py::tuple src_vec = py::reinterpret_borrow<py::tuple>(src);
    const int64_t n = src_vec.size();
    assert(schema.vec().has_value());
    assert(schema.vec()->size() == n);
    for (int64_t i = 0; i < n; ++i) {
      LoadNested(schema.vec()->at(i), src_vec[i], dst);
    }
    return;
  }
  if (py::isinstance<py::list>(src)) {
    const py::list src_vec = py::reinterpret_borrow<py::list>(src);
    const int64_t n = src_vec.size();
    assert(schema.vec().has_value());
    assert(schema.vec()->size() == n);
    for (int64_t i = 0; i < n; ++i) {
      LoadNested(schema.vec()->at(i), src_vec[i], dst);
    }
    return;
  }
  if (py::isinstance<py::dict>(src)) {
    const py::dict src_map = py::reinterpret_borrow<py::dict>(src);
    assert(schema.map().has_value());
    for (const auto& [key, sub] : *schema.map()) {
      LoadNested(sub, src_map[py::str(key)], dst);
    }
    return;
  }
}

}  // namespace

std::pair<int64_t, std::optional<int64_t>> TensorCircularBuffer::Append(
    const py::object& o) {
  if (!initialized_) {
    Init(o);
  }
  const auto ret = Reserve();
  const int64_t index = key_to_index_.at(ret.first);
  const int64_t n = schema_.size();
  std::vector<torch::Tensor> cur(n);
  FlattenNested(schema_, o, cur);
  for (int64_t i = 0; i < n; ++i) {
    data_[i][index] = cur[i];
  }
  return ret;
}

void TensorCircularBuffer::LoadData(const py::object& data,
                                    const py::array_t<int64_t>& keys,
                                    int64_t cursor, int64_t next_key) {
  Reset();
  schema_.FromPython(data, /*packed_input=*/true);
  initialized_ = true;
  data_.resize(schema_.size());
  LoadNested(schema_, data, data_);
  keys_.assign(keys.data(), keys.data() + keys.size());
  const int64_t n = keys_.size();
  for (int64_t i = 0; i < n; ++i) {
    key_to_index_.emplace(keys_[i], i);
  }
  cursor_ = cursor;
  next_key_ = next_key;
}

void TensorCircularBuffer::Init(const py::object& o) {
  schema_.FromPython(o);
  data_.resize(schema_.size());
  ReserveTensors(schema_, capacity_, data_);
  initialized_ = true;
}

py::object TensorCircularBuffer::RecoverNested(
    const std::vector<torch::Tensor>& src) const {
  assert(src.size() == schema_.size());
  return RecoverNestedImpl(schema_, src);
}

std::vector<int64_t> TensorCircularBuffer::AbsoluteIndices(
    int64_t n, const int64_t* indices) const {
  std::vector<int64_t> ret(n);
  for (int64_t i = 0; i < n; ++i) {
    ret[i] = AbsoluteIndex(indices[i]);
  }
  return ret;
}

py::object TensorCircularBuffer::AtImpl(int64_t index) const {
  std::vector<torch::Tensor> ret;
  ret.reserve(data_.size());
  for (const torch::Tensor& tensor : data_) {
    ret.push_back(tensor[index].clone());
  }
  return RecoverNested(ret);
}

void TensorCircularBuffer::BatchedKeyAtImpl(int64_t n, const int64_t* indices,
                                            int64_t* keys) const {
  for (int64_t i = 0; i < n; ++i) {
    keys[i] = keys_.at(indices[i]);
  }
}

py::object TensorCircularBuffer::BatchedValueAtImpl(
    int64_t n, const int64_t* indices) const {
  const torch::Tensor indices_tensor =
      torch::from_blob(const_cast<int64_t*>(indices), {n}, torch::kInt64);
  std::vector<torch::Tensor> ret;
  ret.reserve(data_.size());
  for (const torch::Tensor& tensor : data_) {
    ret.push_back(tensor.index({indices_tensor}));
  }
  return RecoverNested(ret);
}

py::object TensorCircularBuffer::BatchedGetImpl(int64_t n,
                                                const int64_t* keys) const {
  std::vector<int64_t> indices;
  indices.reserve(n);
  for (int64_t i = 0; i < n; ++i) {
    const auto it = key_to_index_.find(keys[i]);
    if (it != key_to_index_.end()) {
      indices.push_back(it->second);
    }
  }
  return BatchedValueAtImpl(indices.size(), indices.data());
}

std::pair<int64_t, std::optional<int64_t>> TensorCircularBuffer::Reserve() {
  const int64_t cur_size = keys_.size();
  const int64_t new_key = next_key_;
  int64_t old_key = -1;
  key_to_index_.emplace(new_key, cursor_);
  if (cur_size < capacity_) {
    keys_.push_back(new_key);
  } else {
    old_key = keys_[cursor_];
    keys_[cursor_] = new_key;
    key_to_index_.erase(old_key);
  }
  NextCursor();
  ++next_key_;
  return std::make_pair(
      new_key, old_key == -1 ? std::nullopt : std::make_optional(old_key));
}

std::pair<std::vector<int64_t>, std::vector<int64_t>>
TensorCircularBuffer::Reserve(int64_t num) {
  std::vector<int64_t> new_keys;
  std::vector<int64_t> old_keys;
  new_keys.reserve(num);
  old_keys.reserve(num);
  for (int64_t i = 0; i < num; ++i) {
    const auto [new_key, old_key] = Reserve();
    new_keys.push_back(new_key);
    if (old_key.has_value()) {
      old_keys.push_back(*old_key);
    }
  }
  return std::make_pair<std::vector<int64_t>, std::vector<int64_t>>(
      std::move(new_keys), std::move(old_keys));
}
void TensorCircularBuffer::InsertImpl(
    const std::vector<int64_t>& keys,
    const std::vector<std::vector<torch::Tensor>>& data, int64_t begin,
    int64_t end) {
  const int64_t m = schema_.size();
  for (int64_t i = begin; i < end; ++i) {
    const auto it = key_to_index_.find(keys[i]);
    if (it == key_to_index_.end()) {
      continue;
    }
    const int64_t index = it->second;
    for (int64_t j = 0; j < m; ++j) {
      data_[j][index] = data[i][j];
    }
  }
}

template <class Sequence>
std::pair<py::array_t<int64_t>, py::array_t<int64_t>>
TensorCircularBuffer::ExtendImpl(const Sequence& src) {
  if (!initialized_) {
    Init(src[0]);
  }
  const int64_t n = src.size();
  const int64_t m = schema_.size();
  const auto [new_keys, old_keys] = Reserve(n);
  std::vector<std::vector<torch::Tensor>> src_data(
      n, std::vector<torch::Tensor>(m));
  for (int64_t i = 0; i < n; ++i) {
    FlattenNested(schema_, src[i], src_data[i]);
  }

  {
    py::gil_scoped_release release;
    const std::vector<int64_t> work_loads = ComputeWorkloads(n, num_threads_);
    const int64_t num_workers = work_loads.size() - 1;
    std::vector<std::thread> threads;
    threads.reserve(num_workers - 1);
    for (int64_t i = 1; i < num_workers; ++i) {
      threads.emplace_back(&TensorCircularBuffer::InsertImpl, this, new_keys,
                           src_data, work_loads[i], work_loads[i + 1]);
    }
    InsertImpl(new_keys, src_data, work_loads[0], work_loads[1]);
    for (auto& job : threads) {
      job.join();
    }
  }
  return std::make_pair<py::array_t<int64_t>, py::array_t<int64_t>>(
      utils::AsNumpyArray<int64_t>(std::move(new_keys)),
      utils::AsNumpyArray<int64_t>(std::move(old_keys)));
}

void DefineTensorCircularBuffer(py::module& m) {
  py::class_<TensorCircularBuffer, std::shared_ptr<TensorCircularBuffer>>(
      m, "TensorCircularBuffer")
      .def(py::init<int64_t, int64_t>(), py::arg("capacity"),
           py::arg("num_threads") = 1)
      .def_property_readonly("capacity", &TensorCircularBuffer::capacity)
      .def_property_readonly("size", &TensorCircularBuffer::Size)
      .def_property_readonly("num_threads", &TensorCircularBuffer::num_threads)
      .def_property_readonly("initialized", &TensorCircularBuffer::initialized)
      .def_property_readonly("cursor", &TensorCircularBuffer::cursor)
      .def_property_readonly("next_key", &TensorCircularBuffer::next_key)
      .def("__len__", &TensorCircularBuffer::Size)
      .def("__getitem__",
           py::overload_cast<int64_t>(&TensorCircularBuffer::At, py::const_))
      .def("__getitem__", py::overload_cast<const py::array_t<int64_t>&>(
                              &TensorCircularBuffer::At, py::const_))
      .def("__getitem__", py::overload_cast<const torch::Tensor&>(
                              &TensorCircularBuffer::At, py::const_))
      .def("reset", &TensorCircularBuffer::Reset)
      .def("clear", &TensorCircularBuffer::Clear)
      .def("front", &TensorCircularBuffer::Front)
      .def("back", &TensorCircularBuffer::Back)
      .def("at",
           py::overload_cast<int64_t>(&TensorCircularBuffer::At, py::const_))
      .def("at", py::overload_cast<const py::array_t<int64_t>&>(
                     &TensorCircularBuffer::At, py::const_))
      .def("at", py::overload_cast<const torch::Tensor&>(
                     &TensorCircularBuffer::At, py::const_))
      .def("get",
           py::overload_cast<int64_t>(&TensorCircularBuffer::Get, py::const_))
      .def("get", py::overload_cast<const py::array_t<int64_t>&>(
                      &TensorCircularBuffer::Get, py::const_))
      .def("get", py::overload_cast<const torch::Tensor&>(
                      &TensorCircularBuffer::Get, py::const_))
      .def("append", &TensorCircularBuffer::Append)
      .def("extend",
           py::overload_cast<const py::tuple&>(&TensorCircularBuffer::Extend))
      .def("extend",
           py::overload_cast<const py::list&>(&TensorCircularBuffer::Extend))
      .def(py::pickle(
          [](const TensorCircularBuffer& buffer) {
            if (buffer.initialized()) {
              return py::make_tuple(buffer.capacity(), buffer.num_threads(),
                                    buffer.initialized(), buffer.DumpData(),
                                    buffer.DumpKeys(), buffer.cursor(),
                                    buffer.next_key());
            } else {
              return py::make_tuple(
                  buffer.capacity(), buffer.num_threads(), buffer.initialized(),
                  py::none(), py::none(), buffer.cursor(), buffer.next_key());
            }
          },
          [](const py::tuple& src) {
            assert(src.size() == 7);
            const int64_t capacity = src[0].cast<int64_t>();
            const int64_t num_threads = src[1].cast<int64_t>();
            const bool initialized = src[2].cast<bool>();
            TensorCircularBuffer buffer(capacity, num_threads);
            if (initialized) {
              const int64_t cursor = src[5].cast<int64_t>();
              const int64_t next_key = src[6].cast<int64_t>();
              buffer.LoadData(src[3], src[4].cast<py::array_t<int64_t>>(),
                              cursor, next_key);
            }
            return buffer;
          }));
}

}  // namespace rlmeta
