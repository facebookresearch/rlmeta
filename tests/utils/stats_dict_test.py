# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np

from rlmeta.utils.stats_dict import StatsDict
from tests.test_utils import TestCaseBase


class StatsDictTest(TestCaseBase):

    def test_add(self) -> None:
        n = 10
        a = np.random.rand(n)
        b = np.random.randn(n)

        d = StatsDict()
        for x, y in zip(a.tolist(), b.tolist()):
            d.add("a", x)
            d.add("b", y)

        self.assertEqual(d["a"].count(), n)
        self.assertEqual(d["a"].mean(), np.mean(a))
        self.assertEqual(d["a"].var(ddof=0), np.var(a, ddof=0))
        self.assertEqual(d["a"].std(ddof=0), np.std(a, ddof=0))
        self.assertEqual(d["a"].min(), np.min(a))
        self.assertEqual(d["a"].max(), np.max(a))

        self.assertEqual(d["b"].count(), n)
        self.assertEqual(d["b"].mean(), np.mean(b))
        self.assertEqual(d["b"].var(ddof=1), np.var(b, ddof=1))
        self.assertEqual(d["b"].std(ddof=1), np.std(b, ddof=1))
        self.assertEqual(d["b"].min(), np.min(b))
        self.assertEqual(d["b"].max(), np.max(b))

    def test_extend(self) -> None:
        n = 10
        a = np.random.rand(n)
        b = np.random.randn(n)

        d = StatsDict()
        for x, y in zip(a.tolist(), b.tolist()):
            d.extend({"a": x, "b": y})

        self.assertEqual(d["a"].count(), n)
        self.assertEqual(d["a"].mean(), np.mean(a))
        self.assertEqual(d["a"].var(ddof=0), np.var(a, ddof=0))
        self.assertEqual(d["a"].std(ddof=0), np.std(a, ddof=0))
        self.assertEqual(d["a"].min(), np.min(a))
        self.assertEqual(d["a"].max(), np.max(a))

        self.assertEqual(d["b"].count(), n)
        self.assertEqual(d["b"].mean(), np.mean(b))
        self.assertEqual(d["b"].var(ddof=1), np.var(b, ddof=1))
        self.assertEqual(d["b"].std(ddof=1), np.std(b, ddof=1))
        self.assertEqual(d["b"].min(), np.min(b))
        self.assertEqual(d["b"].max(), np.max(b))
