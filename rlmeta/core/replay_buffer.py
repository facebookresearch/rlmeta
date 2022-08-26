# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import collections
import time
import logging

from typing import Callable, Optional, Sequence, Tuple, Union
from rich.console import Console
import numpy as np
import torch

import rlmeta.core.remote as remote
import rlmeta.utils.data_utils as data_utils
import rlmeta.utils.nested_utils as nested_utils

from rlmeta.core.launchable import Launchable
from rlmeta.core.server import Server
from rlmeta.core.types import Tensor, NestedTensor
from rlmeta.data import CircularBuffer
from rlmeta.data import SumSegmentTree
from rlmeta.data import TimestampManager
from _rlmeta_extension import Sampler, UniformSampler

console = Console()


class ReplayBuffer(remote.Remotable, Launchable):

    def __init__(self,
                 capacity: int,
                 sampler: Sampler,
                 collate_fn: Optional[Callable[[Sequence[NestedTensor]],
                                               NestedTensor]] = None,
                 identifier: Optional[str] = None):
        remote.Remotable.__init__(self, identifier)

        self._buffer = CircularBuffer(capacity)
        self._sampler = sampler
        if collate_fn is not None:
            self._collate_fn = collate_fn
        else:
            # self._collate_fn = data_utils.stack_tensors
            self._collate_fn = torch.stack

    def __len__(self) -> int:
        return len(self._buffer)

    def __getitem__(self, key: Union[int, Tensor]) -> NestedTensor:
        value = self._buffer[key]
        if not isinstance(key, int):
            value = nested_utils.collate_nested(self._collate_fn, value)
        return value

    @property
    def size(self) -> int:
        return self._buffer.size

    @property
    def capacity(self) -> int:
        return self._buffer.capacity

    @property
    def cursor(self) -> int:
        return self._buffer.cursor

    def init_launching(self) -> None:
        pass

    def init_execution(self) -> None:
        pass

    @remote.remote_method(batch_size=None)
    def get_info(self) -> Tuple[int, int]:
        return self.size, self.capacity

    @remote.remote_method(batch_size=None)
    def append(self, data: NestedTensor, priority: float = 1.0) -> int:
        new_key, old_key = self._buffer.append(data)
        self._sampler.insert(new_key, priority)
        if old_key is not None:
            self._sampler.delete(old_key)
        return new_key

    @remote.remote_method(batch_size=None)
    def extend(self,
               data: Sequence[NestedTensor],
               priorities: Union[float, Tensor] = 1.0) -> None:
        new_keys, old_keys = self._buffer.extend(data)
        self._sampler.insert(
            new_keys,
            priorities.double().numpy()
            if isinstance(priorities, torch.Tensor) else priorities)
        self._sampler.delete(old_keys)
        return torch.from_numpy(new_keys)

    @remote.remote_method(batch_size=None)
    def sample(self, num: int) -> NestedTensor:
        keys, priorities = self._sampler.sample(num)
        values = self.__getitem__(keys)
        return torch.from_numpy(keys), values, torch.from_numpy(priorities)

    @remote.remote_method(batch_size=None)
    def update(self, key: Union[int, Tensor], priority: Union[float,
                                                              Tensor]) -> None:
        self._sampler.update(
            key,
            priority.double().numpy()
            if isinstance(priority, torch.Tensor) else priority)


class RemoteReplayBuffer(remote.Remote):

    def __init__(self,
                 target: ReplayBuffer,
                 server_name: str,
                 server_addr: str,
                 name: Optional[str] = None,
                 prefetch: int = 0,
                 timeout: float = 60) -> None:
        super().__init__(target, server_name, server_addr, name, timeout)
        self._prefetch = prefetch
        self._futures = collections.deque()
        self._server_name = server_name
        self._server_addr = server_addr

    def __repr__(self):
        return (f"RemoteReplayBuffer(server_name={self._server_name}, " +
                f"server_addr={self._server_addr})")

    @property
    def prefetch(self) -> Optional[int]:
        return self._prefetch

    def sample(
        self, batch_size: int
    ) -> Union[NestedTensor, Tuple[NestedTensor, torch.Tensor, torch.Tensor,
                                   torch.Tensor]]:
        if len(self._futures) > 0:
            ret = self._futures.popleft().result()
        else:
            ret = self.client.sync(self.server_name,
                                   self.remote_method_name("sample"),
                                   batch_size)

        while len(self._futures) < self.prefetch:
            fut = self.client.async_(self.server_name,
                                     self.remote_method_name("sample"),
                                     batch_size)
            self._futures.append(fut)

        return ret

    async def async_sample(
        self, batch_size: int
    ) -> Union[NestedTensor, Tuple[NestedTensor, torch.Tensor, torch.Tensor,
                                   torch.Tensor]]:
        if len(self._futures) > 0:
            ret = await self._futures.popleft()
        else:
            ret = await self.client.async_(self.server_name,
                                           self.remote_method_name("sample"),
                                           batch_size)

        while len(self._futures) < self.prefetch:
            fut = self.client.async_(self.server_name,
                                     self.remote_method_name("sample"),
                                     batch_size)
            self._futures.append(fut)

        return ret

    def warm_up(self, learning_starts: Optional[int] = None) -> None:
        # capacity = self.get_capacity()
        size, capacity = self.get_info()
        target_size = capacity
        if learning_starts is not None:
            target_size = min(target_size, learning_starts)
        width = len(str(capacity)) + 1
        # cur_size = self.get_size()
        while size < target_size:
            time.sleep(1)
            # cur_size = self.get_size()
            size, capacity = self.get_info()
            console.log("Warming up replay buffer: " +
                        f"[{size: {width}d} / {capacity} ]")


ReplayBufferLike = Union[ReplayBuffer, RemoteReplayBuffer]


def make_remote_replay_buffer(target: ReplayBuffer,
                              server: Server,
                              name: Optional[str] = None,
                              prefetch: int = 0,
                              timeout: float = 60):
    return RemoteReplayBuffer(target, server.name, server.addr, name, prefetch,
                              timeout)
