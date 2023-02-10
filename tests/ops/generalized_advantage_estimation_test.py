# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from typing import Union

import torch

import rlmeta.ops as ops

from tests.test_utils import TestCaseBase


class GeneralizedAdvantageEstimationTest(TestCaseBase):

    def setUp(self) -> None:
        self.rtol = 1e-6
        self.atol = 1e-6

    def test_gae_with_scalar_parameter(self) -> None:
        n = 100
        gamma = torch.rand(1).item()
        gae_lambda = torch.rand(1).item()

        reward = torch.randn(n)
        value = torch.randn(n + 1)
        gae = ops.generalized_advantage_estimation(reward,
                                                   value,
                                                   gamma,
                                                   gae_lambda,
                                                   terminated=True)
        expected_gae = self._gae(reward,
                                 value,
                                 gamma,
                                 gae_lambda,
                                 terminated=True)
        self.assert_tensor_close(gae,
                                 expected_gae,
                                 rtol=self.rtol,
                                 atol=self.atol)

        reward = torch.randn(n, 1)
        value = torch.randn(n + 1, 1)
        gae = ops.generalized_advantage_estimation(reward,
                                                   value,
                                                   gamma,
                                                   gae_lambda,
                                                   terminated=False)
        expected_gae = self._gae(reward,
                                 value,
                                 gamma,
                                 gae_lambda,
                                 terminated=False)
        self.assert_tensor_close(gae,
                                 expected_gae,
                                 rtol=self.rtol,
                                 atol=self.atol)

    def test_gae_with_tensor_parameter(self) -> None:
        n = 200

        reward = torch.randn(n)
        value = torch.randn(n + 1)
        gamma = torch.rand(1)
        gae_lambda = torch.rand(1)
        gae = ops.generalized_advantage_estimation(reward,
                                                   value,
                                                   gamma,
                                                   gae_lambda,
                                                   terminated=True)
        expected_gae = self._gae(reward,
                                 value,
                                 gamma,
                                 gae_lambda,
                                 terminated=True)
        self.assert_tensor_close(gae,
                                 expected_gae,
                                 rtol=self.rtol,
                                 atol=self.atol)

        reward = torch.randn(n, 1)
        value = torch.randn(n + 1, 1)
        gamma = torch.rand(n, 1)
        gae_lambda = torch.rand(n, 1)
        gae = ops.generalized_advantage_estimation(reward,
                                                   value,
                                                   gamma,
                                                   gae_lambda,
                                                   terminated=False)
        expected_gae = self._gae(reward,
                                 value,
                                 gamma,
                                 gae_lambda,
                                 terminated=False)
        self.assert_tensor_close(gae,
                                 expected_gae,
                                 rtol=self.rtol,
                                 atol=self.atol)

    def _gae(self, reward: torch.Tensor, value: torch.Tensor,
             gamma: Union[float, torch.Tensor], gae_lambda: Union[float,
                                                                  torch.Tensor],
             terminated: bool) -> torch.Tensor:
        n = reward.size(0)

        if terminated:
            assert value.size(0) == n or value.size(0) == n + 1
        else:
            assert value.size(0) == n + 1

        v = 0 if terminated else value[-1]
        adv = 0
        gae = []
        for i in range(n - 1, -1, -1):
            if isinstance(gamma, float):
                gamma_i = gamma
            elif gamma.numel() == 1:
                gamma_i = gamma.item()
            else:
                gamma_i = gamma[i].item()

            if isinstance(gae_lambda, float):
                lambda_i = gae_lambda
            elif gae_lambda.numel() == 1:
                lambda_i = gae_lambda.item()
            else:
                lambda_i = gae_lambda[i].item()

            delta = reward[i] + gamma_i * v - value[i]
            v = value[i]
            adv = delta + gamma_i * lambda_i * adv
            gae.append(adv)

        gae = torch.stack(tuple(reversed(gae)))
        if reward.dim() == 1:
            gae.squeeze_(-1)
        return gae


if __name__ == "__main__":
    unittest.main()
