// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "rlmeta/cc/nested_utils/nested_utils.h"

#include <utility>
#include <vector>

namespace rlmeta {

namespace nested_utils {

namespace {

template <class Function>
void VisitNestedImpl(Function func, const py::object& obj) {
  if (py::isinstance<py::tuple>(obj)) {
    const py::tuple src = py::reinterpret_borrow<py::tuple>(obj);
    for (const auto x : src) {
      VisitNestedImpl(func, py::reinterpret_borrow<py::object>(x));
    }
    return;
  }

  if (py::isinstance<py::list>(obj)) {
    const py::list src = py::reinterpret_borrow<py::list>(obj);
    for (const auto x : src) {
      VisitNestedImpl(func, py::reinterpret_borrow<py::object>(x));
    }
    return;
  }

  if (py::isinstance<py::dict>(obj)) {
    const py::dict src = py::reinterpret_borrow<py::dict>(obj);
    const std::vector<std::string> keys = SortedKeys(src);
    for (const std::string& k : keys) {
      VisitNestedImpl(func,
                      py::reinterpret_borrow<py::object>(src[py::str(k)]));
    }
    return;
  }

  func(obj);
}

template <class Function>
py::object MapNestedImpl(Function func, const py::object& obj) {
  if (py::isinstance<py::tuple>(obj)) {
    const py::tuple src = py::reinterpret_borrow<py::tuple>(obj);
    const int64_t n = src.size();
    py::tuple dst(n);
    for (int64_t i = 0; i < n; ++i) {
      dst[i] = MapNestedImpl(func, src[i]);
    }
    return std::move(dst);
  }

  if (py::isinstance<py::list>(obj)) {
    const py::list src = py::reinterpret_borrow<py::list>(obj);
    const int64_t n = src.size();
    py::list dst(n);
    for (int64_t i = 0; i < n; ++i) {
      dst[i] = MapNestedImpl(func, src[i]);
    }
    return std::move(dst);
  }

  if (py::isinstance<py::dict>(obj)) {
    const py::dict src = py::reinterpret_borrow<py::dict>(obj);
    py::dict dst;
    const std::vector<std::string> keys = SortedKeys(src);
    for (const std::string& k : keys) {
      const py::str key = py::str(k);
      dst[key] =
          MapNestedImpl(func, py::reinterpret_borrow<py::object>(src[key]));
    }
    return std::move(dst);
  }

  return func(obj);
}

template <class Sequence>
py::object CollateNestedImpl(std::function<py::object(const Sequence&)> func,
                             const Sequence& src) {
  const int64_t batch_size = src.size();
  std::vector<py::tuple> flattened;
  size_t index = 0;
  for (int64_t i = 0; i < batch_size; ++i) {
    index = 0;
    VisitNestedImpl(
        [batch_size, i, &flattened, &index](const py::object& obj) {
          py::tuple& cur = index < flattened.size()
                               ? flattened.at(index)
                               : flattened.emplace_back(batch_size);
          cur[i] = obj;
          ++index;
        },
        py::reinterpret_borrow<py::object>(src[i]));
  }
  std::vector<py::object> collated;
  collated.reserve(flattened.size());
  for (const auto& x : flattened) {
    collated.push_back(func(x));
  }
  index = 0;
  return MapNestedImpl(
      [&collated, &index](const py::object& /* obj */) {
        return std::move(collated[index++]);
      },
      src[0]);
}

template <class Sequence>
py::tuple UnbatchSequence(int64_t batch_size, std::vector<py::tuple>& src) {
  py::tuple dst(batch_size);
  const int64_t inner_size = src.size();
  for (int64_t i = 0; i < batch_size; ++i) {
    Sequence cur(inner_size);
    for (int64_t j = 0; j < inner_size; ++j) {
      cur[j] = std::move(src[j][i]);
    }
    dst[i] = std::move(cur);
  }
  return dst;
}

py::tuple UnbatchNestedImpl(std::function<py::tuple(const py::object&)> func,
                            const py::object& obj, int64_t batch_size) {
  if (py::isinstance<py::tuple>(obj)) {
    const py::tuple src = py::reinterpret_borrow<py::tuple>(obj);
    const int64_t n = src.size();
    std::vector<py::tuple> children(n);
    for (int64_t i = 0; i < n; ++i) {
      children[i] = UnbatchNestedImpl(func, src[i], batch_size);
    }
    return UnbatchSequence<py::tuple>(batch_size, children);
  }

  if (py::isinstance<py::list>(obj)) {
    const py::list src = py::reinterpret_borrow<py::list>(obj);
    const int64_t n = src.size();
    std::vector<py::tuple> children(n);
    for (int64_t i = 0; i < n; ++i) {
      children[i] = UnbatchNestedImpl(func, src[i], batch_size);
    }
    return UnbatchSequence<py::list>(batch_size, children);
  }

  if (py::isinstance<py::dict>(obj)) {
    const py::dict src = py::reinterpret_borrow<py::dict>(obj);
    py::tuple dst(batch_size);
    for (int64_t i = 0; i < batch_size; ++i) {
      dst[i] = py::dict();
    }
    const std::vector<std::string> keys = SortedKeys(src);
    for (const std::string& k : keys) {
      const py::str key = py::str(k);
      py::tuple cur = UnbatchNestedImpl(
          func, py::reinterpret_borrow<py::object>(src[key]), batch_size);
      for (int64_t i = 0; i < batch_size; ++i) {
        py::dict y = py::reinterpret_borrow<py::dict>(dst[i]);
        y[key] = cur[i];
      }
    }
    return dst;
  }

  return func(obj);
}

}  // namespace

py::tuple FlattenNested(const py::object& obj) {
  std::vector<py::object> flattened;
  VisitNestedImpl(
      [&flattened](const py::object& obj) { flattened.push_back(obj); }, obj);
  const int64_t n = flattened.size();
  py::tuple ret(n);
  for (int64_t i = 0; i < n; ++i) {
    ret[i] = std::move(flattened[i]);
  }
  return ret;
}

py::object MapNested(std::function<py::object(const py::object&)> func,
                     const py::object& obj) {
  return MapNestedImpl(func, obj);
}

py::object CollateNested(std::function<py::object(const py::tuple&)> func,
                         const py::tuple& src) {
  return CollateNestedImpl<py::tuple>(func, src);
}

py::object CollateNested(std::function<py::object(const py::list&)> func,
                         const py::list& src) {
  return CollateNestedImpl<py::list>(func, src);
}

py::tuple UnbatchNested(std::function<py::tuple(const py::object&)> func,
                        const py::object& obj, int64_t batch_size) {
  return UnbatchNestedImpl(func, obj, batch_size);
}

}  // namespace nested_utils

void DefineNestedUtils(py::module& m) {
  py::module sub = m.def_submodule(
      "nested_utils", "nested_utils submodule of \"_rlmeta_extension\"");

  sub.def("flatten_nested", &nested_utils::FlattenNested)
      .def("map_nested", &nested_utils::MapNested)
      .def("collate_nested",
           py::overload_cast<std::function<py::object(const py::tuple&)>,
                             const py::tuple&>(&nested_utils::CollateNested))
      .def("collate_nested",
           py::overload_cast<std::function<py::object(const py::list&)>,
                             const py::list&>(&nested_utils::CollateNested))
      .def("unbatch_nested", &nested_utils::UnbatchNested);
}

}  // namespace rlmeta
