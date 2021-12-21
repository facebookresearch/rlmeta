# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import torch
import numpy as np

import rlmeta.utils.data_utils as data_utils


class TestCaseBase(unittest.TestCase):
    def assert_tensor_equal(self, x, y):
        self.assertTrue(isinstance(x, type(y)))
        x = data_utils.to_numpy(x)
        y = data_utils.to_numpy(y)
        np.testing.assert_array_equal(x, y)

    def assert_tensor_close(self, x, y, rtol=1e-7, atol=0):
        self.assertTrue(isinstance(x, type(y)))
        x = data_utils.to_numpy(x)
        y = data_utils.to_numpy(y)
        np.testing.assert_allclose(x, y, rtol, atol)
