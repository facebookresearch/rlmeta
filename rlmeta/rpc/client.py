# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pickle

from typing import Any

import grpc
import grpc.experimental

import rlmeta.rpc.rpc_pb2 as rpc_pb2
import rlmeta.rpc.rpc_pb2_grpc as rpc_pb2_grpc


class Client:

    def connect(self, addr: str) -> None:
        # self._channel = grpc.insecure_channel(addr)
        # self._rpc_stub = rpc_pb2_grpc.RpcStub(self._channel)

        self._addr = addr
        self._channel_options = [
            (grpc.experimental.ChannelOptions.SingleThreadedUnaryStream, 1)
        ]

    def rpc(self, function: str, *args, **kwargs) -> Any:
        with grpc.insecure_channel(self._addr,
                                   options=self._channel_options) as channel:
            stub = rpc_pb2_grpc.RpcStub(channel)
            # ret = self._rpc_stub.RemoteCall(
            ret = stub.RemoteCall(
                rpc_pb2.RpcRequest(function=function,
                                   args=pickle.dumps(args),
                                   kwargs=pickle.dumps(kwargs)))
        return pickle.loads(ret.return_value)

    async def async_rpc(self, function: str, *args, **kwargs) -> Any:
        async with grpc.aio.insecure_channel(
                self._addr, options=self._channel_options) as channel:
            stub = rpc_pb2_grpc.RpcStub(channel)
            # ret = await self._rpc_stub.RemoteCall(
            ret = await stub.RemoteCall(
                rpc_pb2.RpcRequest(function=function,
                                   args=pickle.dumps(args),
                                   kwargs=pickle.dumps(kwargs)))
        return pickle.loads(ret.return_value)
