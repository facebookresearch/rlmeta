# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from rlmeta.utils.running_stats import RunningStats
from tests.test_utils import TestCaseBase


class RunningStatsTest(TestCaseBase):

    def setUp(self) -> None:
        self.outer_size = 10
        self.inner_size = (4, 5)
        self.running_stats = RunningStats(self.inner_size)
        self.rtol = 1e-6
        self.atol = 1e-6

    def test_single_update(self) -> None:
        input = torch.rand(self.outer_size, *self.inner_size)
        self.running_stats.reset()
        for x in torch.unbind(input):
            self.running_stats.update(x)
        self.assertEqual(self.running_stats.count(), self.outer_size)
        self.assert_tensor_close(self.running_stats.mean(),
                                 input.mean(dim=0),
                                 rtol=self.rtol,
                                 atol=self.atol)
        self.assert_tensor_close(self.running_stats.var(),
                                 input.var(dim=0, unbiased=False),
                                 rtol=self.rtol,
                                 atol=self.atol)
        self.assert_tensor_close(self.running_stats.var(ddof=1),
                                 input.var(dim=0, unbiased=True),
                                 rtol=self.rtol,
                                 atol=self.atol)
        self.assert_tensor_close(self.running_stats.std(),
                                 input.std(dim=0, unbiased=False),
                                 rtol=self.rtol,
                                 atol=self.atol)
        self.assert_tensor_close(self.running_stats.std(ddof=1),
                                 input.std(dim=0, unbiased=True),
                                 rtol=self.rtol,
                                 atol=self.atol)
        self.assert_tensor_close(self.running_stats.rstd(),
                                 input.std(dim=0, unbiased=False).reciprocal(),
                                 rtol=self.rtol,
                                 atol=self.atol)
        self.assert_tensor_close(self.running_stats.rstd(ddof=1),
                                 input.std(dim=0, unbiased=True).reciprocal(),
                                 rtol=self.rtol,
                                 atol=self.atol)

    def test_batch_update(self) -> None:
        input = torch.rand(self.outer_size, *self.inner_size)
        split_size = [1, 2, 3, 4]
        self.running_stats.reset()
        for x in torch.split(input, split_size):
            self.running_stats.update(x)
        self.assertEqual(self.running_stats.count(), self.outer_size)
        self.assert_tensor_close(self.running_stats.mean(),
                                 input.mean(dim=0),
                                 rtol=self.rtol,
                                 atol=self.atol)
        self.assert_tensor_close(self.running_stats.var(),
                                 input.var(dim=0, unbiased=False),
                                 rtol=self.rtol,
                                 atol=self.atol)
        self.assert_tensor_close(self.running_stats.var(ddof=1),
                                 input.var(dim=0, unbiased=True),
                                 rtol=self.rtol,
                                 atol=self.atol)
        self.assert_tensor_close(self.running_stats.std(),
                                 input.std(dim=0, unbiased=False),
                                 rtol=self.rtol,
                                 atol=self.atol)
        self.assert_tensor_close(self.running_stats.std(ddof=1),
                                 input.std(dim=0, unbiased=True),
                                 rtol=self.rtol,
                                 atol=self.atol)
        self.assert_tensor_close(self.running_stats.rstd(),
                                 input.std(dim=0, unbiased=False).reciprocal(),
                                 rtol=self.rtol,
                                 atol=self.atol)
        self.assert_tensor_close(self.running_stats.rstd(ddof=1),
                                 input.std(dim=0, unbiased=True).reciprocal(),
                                 rtol=self.rtol,
                                 atol=self.atol)


if __name__ == "__main__":
    unittest.main()
