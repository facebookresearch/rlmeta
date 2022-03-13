# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import torch

import rlmeta.core.remote as remote
import rlmeta.utils.remote_utils as remote_utils
from rlmeta.core.server import Server


class RemotableAdder(remote.Remotable):

    @remote.remote_method()
    def add(self, a, b):
        return a + b


class ReplayBufferTest(unittest.TestCase):

    # def test_add(self):
    #     server = Server(name="adder_server", addr="127.0.0.1:4411")
    #     adder = RemotableAdder()
    #     server.add_service(adder)
    #     adder_client = remote_utils.make_remote(adder, server)
    #     server.start()
    #     adder_client.connect()
    #     c = adder_client.add(1, 1)
    #     self.assertEqual(c, 2)
    #     server.terminate()

    def test_add_multiple(self):
        server = Server(name="adder_server", addr="127.0.0.1:4412")
        adder1 = RemotableAdder('a')
        adder2 = RemotableAdder('b')
        self.assertEqual(adder1.identifier, 'a')
        self.assertEqual(adder2.identifier, 'b')
        server.add_service([adder1, adder2])
        adder_client1 = remote_utils.make_remote(adder1, server)
        adder_client2 = remote_utils.make_remote(adder2, server)
        server.start()
        adder_client1.connect()
        c = adder_client1.add(1, 1)
        self.assertEqual(c, 2)
        adder_client2.connect()
        c = adder_client2.add(1, 1)
        self.assertEqual(c, 2)
        server.terminate()

if __name__ == "__main__":
    unittest.main()
