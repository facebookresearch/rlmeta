# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import unittest

from math import prod

import numpy as np
import torch

from rlmeta.data import SumSegmentTree
from tests.test_utils import TestCaseBase


class SumSegmentTreeTest(TestCaseBase):

    def setUp(self) -> None:
        self.size = 100
        self.data = torch.randn(self.size)
        self.segment_tree = SumSegmentTree(self.size, dtype=np.float32)
        self.segment_tree[torch.arange(self.size)] = self.data
        self.query_size = (2, 3, 4)

    def test_at(self) -> None:
        index = torch.randint(self.size, self.query_size)
        value = self.segment_tree[index]
        self.assert_tensor_equal(value, self.data[index])
        value = self.segment_tree.at(index)
        self.assert_tensor_equal(value, self.data[index])

        value = self.segment_tree[index.numpy()]
        self.assert_tensor_equal(value, self.data[index].numpy())
        value = self.segment_tree.at(index.numpy())
        self.assert_tensor_equal(value, self.data[index].numpy())

    def test_update(self) -> None:
        weights = torch.ones(self.size)
        index = weights.multinomial(prod(self.query_size), replacement=False)
        index = index.view(self.query_size)
        origin_value = self.segment_tree[index]

        value = np.random.randn()
        self.segment_tree[index] = value
        self.assert_tensor_equal(self.segment_tree[index],
                                 torch.full(self.query_size, value))
        self.segment_tree[index] = origin_value

        value = np.random.randn()
        self.segment_tree.update(index, value)
        self.assert_tensor_equal(self.segment_tree[index],
                                 torch.full(self.query_size, value))
        self.segment_tree[index] = origin_value

        value = torch.randn(self.query_size)
        self.segment_tree[index] = value
        self.assert_tensor_equal(self.segment_tree[index], value)
        self.segment_tree[index] = origin_value

        value = torch.randn(self.query_size)
        self.segment_tree.update(index, value)
        self.assert_tensor_equal(self.segment_tree[index], value)
        self.segment_tree[index] = origin_value

    def test_masked_update(self) -> None:
        weights = torch.ones(self.size)
        index = weights.multinomial(prod(self.query_size), replacement=False)
        index = index.view(self.query_size)
        origin_value = self.segment_tree[index]
        mask = torch.randint(2, size=self.query_size, dtype=torch.bool)

        value = torch.randn(self.query_size)
        self.segment_tree.update(index, value, mask)
        self.assert_tensor_equal(self.segment_tree[index],
                                 torch.where(mask, value, origin_value))
        self.segment_tree[index] = origin_value

    def test_query(self) -> None:
        a = torch.randint(self.size, self.query_size)
        b = torch.randint(self.size, self.query_size)
        l = torch.minimum(a, b)
        r = torch.maximum(a, b)
        value = self.segment_tree.query(l, r)

        l_list = l.view(-1).tolist()
        r_list = r.view(-1).tolist()
        ret = []
        for (x, y) in zip(l_list, r_list):
            ret.append(self.data[x:y].sum())
        ret = torch.tensor(ret).view(self.query_size)

        self.assert_tensor_close(value, ret, rtol=1e-6, atol=1e-6)

    def test_pickle(self) -> None:
        s = pickle.dumps(self.segment_tree)
        t = pickle.loads(s)
        self.assert_tensor_equal(t[torch.arange(self.size)], self.data)
        for _ in range(10):
            l = np.random.randint(self.size)
            r = np.random.randint(self.size)
            if l > r:
                l, r = r, l
            ret = t.query(l, r)
            ans = self.data[l:r].sum().item()
            self.assertAlmostEqual(ret, ans, places=5)


if __name__ == "__main__":
    unittest.main()
