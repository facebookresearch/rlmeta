# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import pickle

from typing import Any

import grpc
import grpc.experimental

import _rlmeta_extension.rpc as _rpc
import _rlmeta_extension.rpc.rpc_utils as rpc_utils


class Client(_rpc.Client):

    def rpc(self, function: str, *args, **kwargs) -> Any:
        return super().rpc(function, *args, **kwargs)

    def rpc_future(self, function: str, *args, **kwargs) -> _rpc.RpcFuture:
        return super().rpc_future(function, *args, **kwargs)

    def async_rpc(self, function: str, *args, **kwargs) -> Any:
        loop = asyncio.get_running_loop()
        ret = super().rpc_future(function, *args, **kwargs)
        fut = asyncio.Future()
        # loop.call_soon_threadsafe(lambda x: fut.set_result(x.get()), ret)
        loop.call_soon(lambda x: fut.set_result(x.get()), ret)
        return fut
