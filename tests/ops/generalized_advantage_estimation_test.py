# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from typing import Optional, Union

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
        value = torch.randn(n)
        gae = ops.generalized_advantage_estimation(reward, value, gamma,
                                                   gae_lambda)
        expected_gae = self._gae(reward, value, gamma, gae_lambda)
        self.assert_tensor_close(gae,
                                 expected_gae,
                                 rtol=self.rtol,
                                 atol=self.atol)

        reward = torch.randn(n, 1)
        value = torch.randn(n, 1)
        last_v = torch.randn(1)
        gae = ops.generalized_advantage_estimation(reward, value, gamma,
                                                   gae_lambda, last_v)
        expected_gae = self._gae(reward, value, gamma, gae_lambda, last_v)
        self.assert_tensor_close(gae,
                                 expected_gae,
                                 rtol=self.rtol,
                                 atol=self.atol)

    def test_gae_with_tensor_parameter(self) -> None:
        n = 200

        reward = torch.randn(n)
        value = torch.randn(n)
        gamma = torch.rand(1)
        gae_lambda = torch.rand(1)
        gae = ops.generalized_advantage_estimation(reward, value, gamma,
                                                   gae_lambda)
        expected_gae = self._gae(reward, value, gamma, gae_lambda)
        self.assert_tensor_close(gae,
                                 expected_gae,
                                 rtol=self.rtol,
                                 atol=self.atol)

        reward = torch.randn(n, 1)
        value = torch.randn(n, 1)
        gamma = torch.rand(n, 1)
        gae_lambda = torch.rand(n, 1)
        last_v = torch.randn(1)
        gae = ops.generalized_advantage_estimation(reward, value, gamma,
                                                   gae_lambda, last_v)
        expected_gae = self._gae(reward, value, gamma, gae_lambda, last_v)
        self.assert_tensor_close(gae,
                                 expected_gae,
                                 rtol=self.rtol,
                                 atol=self.atol)

    def _gae(self,
             reward: torch.Tensor,
             value: torch.Tensor,
             gamma: Union[float, torch.Tensor],
             gae_lambda: Union[float, torch.Tensor],
             last_v: Optional[torch.Tensor] = None) -> torch.Tensor:
        n = reward.size(0)
        v = torch.zeros(1) if last_v is None else last_v
        adv = torch.zeros(1)
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
