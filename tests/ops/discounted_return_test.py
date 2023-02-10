# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from typing import Union

import torch

import rlmeta.ops as ops

from tests.test_utils import TestCaseBase


class DiscountReturnTest(TestCaseBase):

    def setUp(self) -> None:
        self.rtol = 1e-6
        self.atol = 1e-6

    def test_discounted_return_with_scalar_gamma(self) -> None:
        n = 100
        gamma = torch.rand(1).item()

        reward = torch.randn(n)
        g = ops.discounted_return(reward, gamma)
        expected_g = self._discounted_return(reward, gamma)
        self.assert_tensor_close(g, expected_g, rtol=self.rtol, atol=self.atol)

        reward = torch.randn(n, 1)
        g = ops.discounted_return(reward, gamma)
        expected_g = self._discounted_return(reward, gamma)
        self.assert_tensor_close(g, expected_g, rtol=self.rtol, atol=self.atol)

    def test_discounted_return_with_tensor_gamma(self) -> None:
        n = 200

        gamma = torch.rand(1)
        reward = torch.randn(n)
        g = ops.discounted_return(reward, gamma)
        expected_g = self._discounted_return(reward, gamma)
        self.assert_tensor_close(g, expected_g, rtol=self.rtol, atol=self.atol)

        gamma = torch.rand(n)
        reward = torch.randn(n, 1)
        g = ops.discounted_return(reward, gamma)
        expected_g = self._discounted_return(reward, gamma)
        self.assert_tensor_close(g, expected_g, rtol=self.rtol, atol=self.atol)

    def _discounted_return(self, reward: torch.Tensor,
                           gamma: Union[float, torch.Tensor]) -> torch.Tensor:
        n = reward.size(0)
        g = torch.zeros(1)
        ret = []
        for i in range(n - 1, -1, -1):
            if isinstance(gamma, float):
                gamma_i = gamma
            elif gamma.numel() == 1:
                gamma_i = gamma.item()
            else:
                gamma_i = gamma[i].item()
            g = reward[i] + gamma_i * g
            ret.append(g)
        ret = torch.stack(tuple(reversed(ret)))
        if reward.dim() == 1:
            ret.squeeze_(-1)
        return ret


if __name__ == "__main__":
    unittest.main()
