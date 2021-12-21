# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math

from tabulate import tabulate
from typing import Dict, Optional


class StatsItem:
    def __init__(self, key: Optional[str] = None) -> None:
        self._key = key
        self.reset()

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
        return self._m2 / (self._m0 - ddof) if self._m0 > 1 else float("nan")

    def std(self, ddof: int = 0) -> float:
        return math.sqrt(self.var(ddof))

    def min(self) -> float:
        return self._min_val

    def max(self) -> float:
        return self._max_val


class StatsDict:
    def __init__(self) -> None:
        self._dict = {}

    def __getitem__(self, key: str) -> StatsItem:
        return self._dict[key]

    def reset(self):
        self._dict.clear()

    def add(self, k: str, v: float) -> None:
        if k in self._dict:
            self._dict[k].add(v)
        else:
            item = StatsItem(k)
            item.add(v)
            self._dict[k] = item

    def add_dict(self, d: Dict[str, float]) -> None:
        for k, v in d.items():
            self.add(k, v)

    def update(self, stats: StatsDict) -> None:
        self._dict.update(stats._dict)

    def table(self, info: Optional[str] = None) -> str:
        h = ["info"] if info is not None else []
        h += ["key", "mean", "std", "min", "max", "count"]
        t = []
        for k, v in self._dict.items():
            row = [info] if info is not None else []
            row += [k, v.mean(), v.std(), v.min(), v.max(), v.count()]
            t.append(row)
        return tabulate(t,
                        h,
                        numalign="right",
                        stralign="right",
                        floatfmt=".8f")
