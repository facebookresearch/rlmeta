# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import math

from typing import Dict, Optional

from tabulate import tabulate


class StatsItem:

    def __init__(self,
                 key: Optional[str] = None,
                 val: Optional[float] = None) -> None:
        self._key = key
        self.reset()
        if val is not None:
            self.add(val)

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> StatsItem:
        ret = cls(key=data.get("key", None))
        ret._m0 = data.get("count", 0)
        ret._m1 = data.get("mean", 0.0)
        std = data.get("std", 0.0)
        ret._m2 = std * std * ret._m0
        ret._min_val = data.get("min", float("inf"))
        ret._max_val = data.get("max", float("-inf"))
        return ret

    @property
    def key(self) -> str:
        return self._key

    def reset(self):
        self._m0 = 0
        self._m1 = 0.0
        self._m2 = 0.0

        self._min_val = float("inf")
        self._max_val = float("-inf")

    def add(self, v: float) -> None:
        # Welford algorithm.
        self._m0 += 1
        delta = v - self._m1
        self._m1 += delta / self._m0
        self._m2 += delta * (v - self._m1)

        self._min_val = min(self._min_val, v)
        self._max_val = max(self._max_val, v)

    def count(self) -> int:
        return self._m0

    def mean(self) -> float:
        return self._m1

    def var(self, ddof: int = 0) -> float:
        return self._m2 / (self._m0 - ddof)

    def std(self, ddof: int = 0) -> float:
        return math.sqrt(self.var(ddof))

    def min(self) -> float:
        return self._min_val

    def max(self) -> float:
        return self._max_val

    def dict(self) -> Dict[str, float]:
        ret = {
            "mean": self.mean(),
            "std": self.std(),
            "min": self.min(),
            "max": self.max(),
            "count": self.count(),
        }
        if self.key is not None:
            ret["key"] = self.key
        return ret


class StatsDict:

    def __init__(self) -> None:
        self._dict = {}

    def __getitem__(self, key: str) -> StatsItem:
        return self._dict[key]

    @classmethod
    def from_dict(cls, data: Dict[str, Dict[str, float]]) -> StatsDict:
        ret = cls()
        ret._dict = {k: StatsItem.from_dict(v) for k, v in data.items()}
        return ret

    def reset(self):
        self._dict.clear()

    def add(self, k: str, v: float) -> None:
        if k in self._dict:
            self._dict[k].add(v)
        else:
            self._dict[k] = StatsItem(k, v)

    def extend(self, d: Dict[str, float]) -> None:
        for k, v in d.items():
            self.add(k, v)

    def update(self, stats: StatsDict) -> None:
        self._dict.update(stats._dict)

    def dict(self) -> Dict[str, Dict[str, float]]:
        return {k: v.dict() for k, v in self._dict.items()}

    def json(self, info: Optional[str] = None, **kwargs) -> str:
        data = self.dict()
        if info is not None:
            data["info"] = info
        data.update(kwargs)
        return json.dumps(data)

    def table(self, info: Optional[str] = None, **kwargs) -> str:
        if info is None:
            head = ["key", "mean", "std", "min", "max", "count"]
        else:
            head = ["info", "key", "mean", "std", "min", "max", "count"]

        data = []
        for k, v in self._dict.items():
            if info is None:
                row = [k, v.mean(), v.std(), v.min(), v.max(), v.count()]
            else:
                row = [info, k, v.mean(), v.std(), v.min(), v.max(), v.count()]
            data.append(row)
        for k, v in kwargs.items():
            if info is None:
                row = [k, v, 0.0, v, v, 1]
            else:
                row = [info, k, v, 0.0, v, v, 1]
            data.append(row)

        return tabulate(data,
                        head,
                        numalign="right",
                        stralign="right",
                        floatfmt=".8f")
