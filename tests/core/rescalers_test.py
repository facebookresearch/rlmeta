# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import torch

from rlmeta.core.rescalers import MomentsRescaler, RMSRescaler, SqrtRescaler
from tests.test_utils import TestCaseBase


class RescalerTest(TestCaseBase):

    def setUp(self) -> None:
        self.size = (4, 5)
        self.rtol = 1e-5
        self.atol = 1e-5

    def test_rms_rescaler(self) -> None:
        rms_rescaler = RMSRescaler(self.size)

        batch_size = np.random.randint(low=1, high=10)
        data = torch.rand(batch_size, *self.size)
        for x in torch.unbind(data):
            rms_rescaler.update(x)

        x = torch.rand(*self.size)
        y = rms_rescaler.rescale(x)
        y_expected = x / data.square().mean(dim=0).sqrt()
        self.assert_tensor_close(y, y_expected, rtol=self.rtol, atol=self.atol)
        self.assert_tensor_close(rms_rescaler.recover(y),
                                 x,
                                 rtol=self.rtol,
                                 atol=self.atol)

    def test_norm_rescaler(self) -> None:
        norm_rescaler = MomentsRescaler(self.size)

        batch_size = np.random.randint(low=1, high=10)
        data = torch.rand(batch_size, *self.size)
        for x in torch.unbind(data):
            norm_rescaler.update(x)

        x = torch.rand(*self.size)
        y = norm_rescaler.rescale(x)
        if batch_size == 1:
            y_expected = x
        else:
            y_expected = (x - data.mean(dim=0)) / data.std(dim=0,
                                                           unbiased=False)
        self.assert_tensor_close(y, y_expected, rtol=self.rtol, atol=self.atol)
        self.assert_tensor_close(norm_rescaler.recover(y),
                                 x,
                                 rtol=self.rtol,
                                 atol=self.atol)

    def test_sqrt_rescaler(self) -> None:
        eps = np.random.choice([0.0, 1e-5, 1e-3, 2e-2, 0.5])
        sqrt_rescaler = SqrtRescaler(eps)

        x = torch.randn(*self.size, dtype=torch.float64)
        y = sqrt_rescaler.rescale(x)
        y_expected = x.sign() * ((x.abs() + 1).sqrt() - 1) + eps * x
        self.assert_tensor_close(y, y_expected, rtol=self.rtol, atol=self.atol)
        self.assert_tensor_close(sqrt_rescaler.recover(y),
                                 x,
                                 rtol=self.rtol,
                                 atol=self.atol)


if __name__ == "__main__":
    unittest.main()
