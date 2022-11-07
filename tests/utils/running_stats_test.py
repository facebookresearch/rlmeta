# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from rlmeta.utils.running_stats import RunningMoments, RunningRMS
from tests.test_utils import TestCaseBase


class RunningRMSTest(TestCaseBase):

    def setUp(self) -> None:
        self.outer_size = 10
        self.inner_size = (4, 5)
        self.running_rms = RunningRMS(self.inner_size)
        self.rtol = 1e-6
        self.atol = 1e-6

    def test_single_update(self) -> None:
        input = torch.rand(self.outer_size, *self.inner_size)
        self.running_rms.reset()
        for x in torch.unbind(input):
            self.running_rms.update(x)
        self._verify_running_rms(input)

    def test_batch_update(self) -> None:
        input = torch.rand(self.outer_size, *self.inner_size)
        split_size = [1, 2, 3, 4]
        self.running_rms.reset()
        for x in torch.split(input, split_size):
            self.running_rms.update(x)
        self._verify_running_rms(input)

    def _verify_running_rms(self, input: torch.Tensor) -> None:
        self.assert_tensor_equal(self.running_rms.count(),
                                 torch.tensor([self.outer_size]))
        self.assert_tensor_close(self.running_rms.mean_square(),
                                 input.square().mean(dim=0),
                                 rtol=self.rtol,
                                 atol=self.atol)
        self.assert_tensor_close(self.running_rms.rms(),
                                 input.square().mean(dim=0).sqrt(),
                                 rtol=self.rtol,
                                 atol=self.atol)
        self.assert_tensor_close(self.running_rms.rrms(),
                                 input.square().mean(dim=0).rsqrt(),
                                 rtol=self.rtol,
                                 atol=self.atol)


class RunningMomentsTest(TestCaseBase):

    def setUp(self) -> None:
        self.outer_size = 10
        self.inner_size = (4, 5)
        self.running_moments = RunningMoments(self.inner_size)
        self.rtol = 1e-6
        self.atol = 1e-6

    def test_single_update(self) -> None:
        input = torch.rand(self.outer_size, *self.inner_size)
        self.running_moments.reset()
        for x in torch.unbind(input):
            self.running_moments.update(x)
        self._verify_running_moments(input)

    def test_batch_update(self) -> None:
        input = torch.rand(self.outer_size, *self.inner_size)
        split_size = [1, 2, 3, 4]
        self.running_moments.reset()
        for x in torch.split(input, split_size):
            self.running_moments.update(x)
        self._verify_running_moments(input)

    def _verify_running_moments(self, input: torch.Tensor) -> None:
        self.assert_tensor_equal(self.running_moments.count(),
                                 torch.tensor([self.outer_size]))
        self.assert_tensor_close(self.running_moments.mean(),
                                 input.mean(dim=0),
                                 rtol=self.rtol,
                                 atol=self.atol)
        self.assert_tensor_close(self.running_moments.var(),
                                 input.var(dim=0, unbiased=False),
                                 rtol=self.rtol,
                                 atol=self.atol)
        self.assert_tensor_close(self.running_moments.var(ddof=1),
                                 input.var(dim=0, unbiased=True),
                                 rtol=self.rtol,
                                 atol=self.atol)
        self.assert_tensor_close(self.running_moments.std(),
                                 input.std(dim=0, unbiased=False),
                                 rtol=self.rtol,
                                 atol=self.atol)
        self.assert_tensor_close(self.running_moments.std(ddof=1),
                                 input.std(dim=0, unbiased=True),
                                 rtol=self.rtol,
                                 atol=self.atol)
        self.assert_tensor_close(self.running_moments.rstd(),
                                 input.std(dim=0, unbiased=False).reciprocal(),
                                 rtol=self.rtol,
                                 atol=self.atol)
        self.assert_tensor_close(self.running_moments.rstd(ddof=1),
                                 input.std(dim=0, unbiased=True).reciprocal(),
                                 rtol=self.rtol,
                                 atol=self.atol)


if __name__ == "__main__":
    unittest.main()
