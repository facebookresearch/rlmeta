# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

import rlmeta.utils.data_utils as data_utils

from rlmeta.core.replay_buffer import ReplayBuffer
from rlmeta.samplers import UniformSampler, PrioritizedSampler
from rlmeta.storage import CircularBuffer, TensorCircularBuffer
from tests.test_utils import TestCaseBase


class ReplayBufferTest(TestCaseBase):

    def setUp(self):
        self.size = 8
        self.batch_size = 5
        self.hidden_dim = 4

        self.replay_buffer = ReplayBuffer(
            CircularBuffer(self.size, collate_fn=torch.stack), UniformSampler())
        self.flatten_data = dict(obs=torch.randn(self.batch_size,
                                                 self.hidden_dim),
                                 rew=torch.randn(self.batch_size))
        self.data = data_utils.unstack_fields(self.flatten_data,
                                              self.batch_size)

    def test_extend(self):
        self.replay_buffer.clear()

        keys = self.replay_buffer.extend(self.data)
        expected_keys = torch.arange(self.batch_size)
        self.assertEqual(len(self.replay_buffer), self.batch_size)
        self.assert_tensor_equal(keys, expected_keys)
        data = self.replay_buffer[keys]
        self.assertEqual(data.keys(), self.flatten_data.keys())
        for k, v in data.items():
            self.assert_tensor_equal(v, self.flatten_data[k])

        keys = self.replay_buffer.extend(self.data)
        self.assertEqual(len(self.replay_buffer), self.size)
        self.assert_tensor_equal(keys, expected_keys + self.batch_size)
        data = self.replay_buffer[keys]
        self.assertEqual(data.keys(), self.flatten_data.keys())
        for k, v in data.items():
            self.assert_tensor_equal(v, self.flatten_data[k])

    def test_sample(self):
        self.replay_buffer.clear()
        self.replay_buffer.extend(self.data)

        num_samples = 10000
        prob = 1.0 / self.batch_size
        keys, _, probs = self.replay_buffer.sample(num_samples)
        self.assert_tensor_equal(
            probs, torch.full((num_samples,), prob, dtype=torch.float64))
        actual_probs = torch.bincount(keys).float() / num_samples
        self.assert_tensor_close(actual_probs,
                                 torch.full((self.batch_size,), prob),
                                 atol=0.05)


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

    def test_extend(self):
        replay_buffer = ReplayBuffer(TensorCircularBuffer(self.size),
                                     PrioritizedSampler(priority_exponent=0.6))
        keys = replay_buffer.extend(self.data)
        expected_keys = torch.arange(self.batch_size)
        self.assertEqual(len(replay_buffer), self.batch_size)
        self.assert_tensor_equal(keys, expected_keys)
        data = replay_buffer[keys]
        self.assertEqual(data.keys(), self.flatten_data.keys())
        for k, v in data.items():
            self.assert_tensor_equal(v, self.flatten_data[k])

        keys = replay_buffer.extend(self.data)
        self.assertEqual(len(replay_buffer), self.size)
        self.assert_tensor_equal(keys, expected_keys + self.batch_size)
        data = replay_buffer[keys]
        self.assertEqual(data.keys(), self.flatten_data.keys())
        for k, v in data.items():
            self.assert_tensor_equal(v, self.flatten_data[k])

    def test_sample(self):
        replay_buffer = ReplayBuffer(TensorCircularBuffer(self.size),
                                     PrioritizedSampler(priority_exponent=1.0))
        priorities = torch.rand((self.batch_size,)) * 10
        expected_probs = priorities / priorities.sum()
        replay_buffer.extend(self.data, priorities=priorities)

        num_samples = 1000
        keys, _, probs = replay_buffer.sample(num_samples)

        actual_probs = torch.bincount(keys).float() / num_samples
        self.assert_tensor_close(probs, expected_probs[keys], rtol=1e-6)
        self.assert_tensor_close(actual_probs, expected_probs, atol=0.05)

    def test_update(self):
        alpha = 0.6
        replay_buffer = ReplayBuffer(
            TensorCircularBuffer(self.size),
            PrioritizedSampler(priority_exponent=alpha))
        priorities = torch.rand((self.batch_size,)) * 10
        keys = replay_buffer.extend(self.data, priorities=priorities)
        priorities = torch.rand((self.batch_size,)) * 10
        expected_probs = priorities.pow(alpha)
        expected_probs.div_(expected_probs.sum())
        replay_buffer.update(keys, priorities)

        num_samples = 100
        keys, _, probs = replay_buffer.sample(num_samples)
        self.assert_tensor_close(probs, expected_probs[keys], rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
