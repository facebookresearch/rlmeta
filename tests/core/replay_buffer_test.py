# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import torch

import rlmeta.utils.data_utils as data_utils

from rlmeta.core.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from tests.test_utils import TestCaseBase


class ReplayBufferTest(TestCaseBase):

    def setUp(self):
        self.size = 8
        self.batch_size = 5
        self.hidden_dim = 4

        self.flatten_data = dict(obs=torch.randn(self.batch_size,
                                                 self.hidden_dim),
                                 rew=torch.randn(self.batch_size))
        self.data = data_utils.unstack_fields(self.flatten_data,
                                              self.batch_size)

    def test_extend(self):
        replay_buffer = ReplayBuffer(self.size)

        index = replay_buffer._extend(self.data)
        expected_index = np.arange(self.batch_size)
        self.assertEqual(replay_buffer.cursor, self.batch_size % self.size)
        self.assertEqual(len(replay_buffer), self.batch_size)
        self.assert_tensor_equal(index, expected_index)
        data = replay_buffer[index]
        self.assertEqual(data.keys(), self.flatten_data.keys())
        for k, v in data.items():
            self.assert_tensor_equal(v, self.flatten_data[k])

        index = replay_buffer._extend(self.data)
        d = self.size - self.batch_size
        expected_index = np.empty(self.batch_size, dtype=np.int64)
        for i in range(d):
            expected_index[i] = self.batch_size + i
        for i in range(self.batch_size - d):
            expected_index[d + i] = i
        self.assertEqual(replay_buffer.cursor, self.batch_size * 2 % self.size)
        self.assertEqual(len(replay_buffer), self.size)
        self.assert_tensor_equal(index, expected_index)
        if d > 0:
            index1 = index[:d]
            data1 = replay_buffer[index1]
            self.assertEqual(data1.keys(), self.flatten_data.keys())
            for k, v in data1.items():
                self.assert_tensor_equal(v, self.flatten_data[k][:d])
        if d < self.batch_size:
            index2 = index[d:]
            data2 = replay_buffer[index2]
            for k, v in data2.items():
                self.assert_tensor_equal(v, self.flatten_data[k][d:])

    def test_sample(self):
        replay_buffer = ReplayBuffer(self.size)
        # Only add 1 data into replay_buffer,
        # so the sample output should be the same.
        replay_buffer.append(self.data[0])
        batch = replay_buffer.sample(self.batch_size)
        self.assertEqual(data_utils.size(batch["obs"])[0], self.batch_size)
        self.assertEqual(data_utils.size(batch["rew"])[0], self.batch_size)
        self.assertEqual(batch.keys(), self.flatten_data.keys())
        for k, v in batch.items():
            for i in range(self.batch_size):
                self.assert_tensor_equal(v[i], self.flatten_data[k][0])


class PrioritizedReplayBufferTest(TestCaseBase):

    def setUp(self):
        self.size = 8
        self.batch_size = 5
        self.hidden_dim = 4

        self.flatten_data = dict(obs=torch.randn(self.batch_size,
                                                 self.hidden_dim),
                                 rew=torch.randn(self.batch_size))
        self.data = data_utils.unstack_fields(self.flatten_data,
                                              self.batch_size)

        self.alpha = 0.8
        self.beta = 0.5

        self.rtol = 1e-6
        self.atol = 1e-6

    def test_extend(self):
        replay_buffer = PrioritizedReplayBuffer(self.size, self.alpha,
                                                self.beta)
        index = replay_buffer._extend(self.data)
        expected_index = np.arange(self.batch_size)
        expected_weights = torch.ones(self.batch_size)
        self.assertEqual(len(replay_buffer), self.batch_size)
        self.assert_tensor_equal(index, expected_index)
        data, weights = replay_buffer[index]
        self.assertEqual(data.keys(), self.flatten_data.keys())
        for k, v in data.items():
            self.assert_tensor_equal(v, self.flatten_data[k])
        self.assert_tensor_equal(weights, expected_weights)

        index = replay_buffer._extend(self.data)
        d = self.size - self.batch_size
        expected_index = np.empty(self.batch_size, dtype=np.int64)
        expected_weights = torch.ones(self.batch_size)
        for i in range(d):
            expected_index[i] = self.batch_size + i
        for i in range(self.batch_size - d):
            expected_index[d + i] = i
        self.assertEqual(len(replay_buffer), self.size)
        self.assert_tensor_equal(index, expected_index)
        if d > 0:
            index1 = index[:d]
            data1, weights1 = replay_buffer[index1]
            self.assertEqual(data1.keys(), self.flatten_data.keys())
            for k, v in data1.items():
                self.assert_tensor_equal(v, self.flatten_data[k][:d])
            self.assert_tensor_equal(weights1, expected_weights[:d])
        if d < self.batch_size:
            index2 = index[d:]
            data2, weights2 = replay_buffer[index2]
            for k, v in data2.items():
                self.assert_tensor_equal(v, self.flatten_data[k][d:])
            self.assert_tensor_equal(weights2, expected_weights[d:])

    def test_update_priority(self):
        replay_buffer = PrioritizedReplayBuffer(self.size, self.alpha,
                                                self.beta)
        replay_buffer._extend(self.data)
        replay_buffer._extend(self.data)

        index = torch.arange(self.size)
        weight = torch.rand(self.size) * 3.0
        expected_weights = torch.pow(weight, self.alpha)
        expected_weights = torch.pow(expected_weights / expected_weights.min(),
                                     -self.beta)
        expected_max_weight = weight.max()
        replay_buffer.update_priority(index, weight)
        _, cur_weights = replay_buffer[index]
        self.assert_tensor_close(cur_weights, expected_weights, self.rtol,
                                 self.atol)
        self.assertAlmostEqual(replay_buffer.max_priority,
                               expected_max_weight,
                               delta=self.atol)

    def test_sample(self):
        replay_buffer = PrioritizedReplayBuffer(self.size, self.alpha,
                                                self.beta)
        replay_buffer._extend(self.data)
        replay_buffer._extend(self.data)

        index = torch.arange(self.size)
        weight = torch.rand(self.size)
        expected_weight = torch.pow(weight, self.alpha)

        replay_buffer.update_priority(index, weight)
        _, cur_weight, cur_index, _ = replay_buffer.sample(self.batch_size)
        expected_weight = expected_weight[cur_index]
        expected_weight = (expected_weight /
                           expected_weight.min()).pow(-self.beta)
        self.assert_tensor_close(cur_weight, expected_weight, self.rtol,
                                 self.atol)


if __name__ == "__main__":
    unittest.main()
