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

    def setUp(self) -> None:
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

    def test_extend(self) -> None:
        self.replay_buffer.reset()

        keys = self.replay_buffer.extend(self.data)
        expected_keys = torch.arange(self.batch_size)
        self.assertEqual(len(self.replay_buffer), self.batch_size)
        self.assert_tensor_equal(keys, expected_keys)
        data = self.replay_buffer.get(keys)
        self.assertEqual(data.keys(), self.flatten_data.keys())
        for k, v in data.items():
            self.assert_tensor_equal(v, self.flatten_data[k])

        keys = self.replay_buffer.extend(self.data)
        self.assertEqual(len(self.replay_buffer), self.size)
        self.assert_tensor_equal(keys, expected_keys + self.batch_size)
        data = self.replay_buffer.get(keys)
        self.assertEqual(data.keys(), self.flatten_data.keys())
        for k, v in data.items():
            self.assert_tensor_equal(v, self.flatten_data[k])

    def test_sample(self) -> None:
        self.replay_buffer.reset()
        self.replay_buffer.extend(self.data)

        prob = 1.0 / self.batch_size

        num_samples = self.batch_size
        keys, _, probs = self.replay_buffer.sample(num_samples)
        expected_probs = torch.full_like(probs, prob)
        self.assert_tensor_equal(probs, expected_probs)
        count = torch.bincount(keys)
        self.assertEqual(count.max().item(), 1)
        count = torch.zeros(self.batch_size, dtype=torch.int64)
        for _ in range(20000):
            keys, _, _ = self.replay_buffer.sample(3)
            count[keys] += 1
        actual_probs = count / count.sum()
        expected_probs = torch.full_like(actual_probs, prob)
        self.assert_tensor_close(actual_probs, expected_probs, atol=0.05)

        # Test sample with replacement.
        num_samples = 20000
        keys, _, probs = self.replay_buffer.sample(num_samples,
                                                   replacement=True)
        self.assert_tensor_equal(
            probs, torch.full((num_samples,), prob, dtype=torch.float64))
        actual_probs = torch.bincount(keys).float() / num_samples
        expected_probs = torch.full_like(actual_probs, prob)
        self.assert_tensor_close(actual_probs, expected_probs, atol=0.05)

    def test_clear(self) -> None:
        self.replay_buffer.reset()

        self.replay_buffer.extend(self.data)
        self.assertEqual(len(self.replay_buffer), len(self.data))
        self.replay_buffer.clear()
        self.assertEqual(len(self.replay_buffer), 0)
        self.replay_buffer.extend(self.data)
        self.assertEqual(len(self.replay_buffer), len(self.data))


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
        data = replay_buffer.get(keys)
        self.assertEqual(data.keys(), self.flatten_data.keys())
        for k, v in data.items():
            self.assert_tensor_equal(v, self.flatten_data[k])

        keys = replay_buffer.extend(self.data)
        self.assertEqual(len(replay_buffer), self.size)
        self.assert_tensor_equal(keys, expected_keys + self.batch_size)
        data = replay_buffer.get(keys)
        self.assertEqual(data.keys(), self.flatten_data.keys())
        for k, v in data.items():
            self.assert_tensor_equal(v, self.flatten_data[k])

    def test_sample(self):
        replay_buffer = ReplayBuffer(TensorCircularBuffer(self.size),
                                     PrioritizedSampler(priority_exponent=1.0))
        priorities = torch.rand((self.batch_size,)) * 10
        expected_probs = priorities / priorities.sum()
        replay_buffer.extend(self.data, priorities=priorities)

        # Test sample without replacement
        num_samples = self.batch_size
        keys, _, probs = replay_buffer.sample(num_samples)
        self.assert_tensor_close(probs,
                                 expected_probs[keys],
                                 rtol=1e-6,
                                 atol=1e-6)
        count = torch.bincount(keys)
        self.assertEqual(count.max().item(), 1)
        count = torch.zeros(self.batch_size, dtype=torch.int64)
        for _ in range(100000):
            keys, _, _ = replay_buffer.sample(3)
            count[keys] += 1
        actual_probs = count / count.sum()
        self.assert_tensor_close(actual_probs, expected_probs, atol=0.1)

        # Test sample with replacement.
        num_samples = 100000
        keys, _, probs = replay_buffer.sample(num_samples, replacement=True)
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
        keys, _, probs = replay_buffer.sample(num_samples, replacement=True)
        self.assert_tensor_close(probs, expected_probs[keys], rtol=1e-6)

    def test_reset(self) -> None:
        replay_buffer = ReplayBuffer(TensorCircularBuffer(self.size),
                                     PrioritizedSampler(priority_exponent=0.6))
        replay_buffer.extend(self.data)
        self.assertEqual(len(replay_buffer), len(self.data))
        replay_buffer.reset()
        self.assertEqual(len(replay_buffer), 0)
        self.assertFalse(replay_buffer._storage._impl.initialized)
        replay_buffer.extend(self.data)
        self.assertEqual(len(replay_buffer), len(self.data))

    def test_clear(self) -> None:
        replay_buffer = ReplayBuffer(TensorCircularBuffer(self.size),
                                     PrioritizedSampler(priority_exponent=0.6))

        replay_buffer.extend(self.data)
        self.assertEqual(len(replay_buffer), len(self.data))
        replay_buffer.clear()
        self.assertEqual(len(replay_buffer), 0)
        self.assertTrue(replay_buffer._storage._impl.initialized)
        replay_buffer.extend(self.data)
        self.assertEqual(len(replay_buffer), len(self.data))


if __name__ == "__main__":
    unittest.main()
