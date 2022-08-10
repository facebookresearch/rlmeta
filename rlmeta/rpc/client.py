# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import pickle

from typing import Any

import grpc
import grpc.experimental

import rlmeta.rpc.rpc_pb2 as rpc_pb2
import rlmeta.rpc.rpc_pb2_grpc as rpc_pb2_grpc
import _rlmeta_extension.rpc as _rpc
import _rlmeta_extension.rpc.rpc_utils as _rpc_utils


class Client(_rpc.Client):

    def __init__(self, py_aio_client: bool = True) -> None:
        super().__init__()
        self._addr = None
        self._timeout = None
        self._options = None
        self._connected = False
        self._py_aio_client = py_aio_client

    def connect(self, addr: str, timeout: int = 60) -> None:
        super().connect(addr, timeout)
        self._addr = addr
        self._timeout = timeout
        self._options = [
            (grpc.experimental.ChannelOptions.SingleThreadedUnaryStream, 1),
            ("grpc.enable_http_proxy", 0),
            ("grpc.max_send_message_length", -1),
            ("grpc.max_receive_message_length", -1),
        ]
        self._connected = True

    def rpc(self, function: str, *args, **kwargs) -> Any:
        return super().rpc(function, *args, **kwargs)

    def rpc_future(self, function: str, *args, **kwargs) -> _rpc.RpcFuture:
        return super().rpc_future(function, *args, **kwargs)

    async def async_rpc(self, function: str, *args, **kwargs) -> Any:
        if self._py_aio_client:
            return await self._async_rpc_py(function, *args, **kwargs)
        else:
            return await self._async_rpc_cc(function, *args, **kwargs)

    async def _async_rpc_cc(self, function: str, *args, **kwargs) -> Any:
        loop = asyncio.get_running_loop()
        ret = super().rpc_future(function, *args, **kwargs)
        fut = asyncio.Future()
        # loop.call_soon_threadsafe(lambda x: fut.set_result(x.get()), ret)
        loop.call_soon(lambda x: fut.set_result(x.get()), ret)
        return await fut

    # TODO: Add efficient native asyncio support in C++ client.
    async def _async_rpc_py(self, function: str, *args, **kwargs) -> Any:
        async with grpc.aio.insecure_channel(target=self._addr,
                                             options=self._options) as channel:
            stub = rpc_pb2_grpc.RpcStub(channel)
            response = await stub.PyRemoteCall(
                rpc_pb2.PyRpcRequest(function=function,
                                     args=_rpc_utils.dumps(args),
                                     kwargs=_rpc_utils.dumps(kwargs)))
            return _rpc_utils.loads(response.return_value)
