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
from rlmeta.storage import Storage
from rlmeta.samplers import Sampler

console = Console()

# The design of ReplayBuffer is inspired from DeepMind's Reverb project.
#
# https://github.com/deepmind/reverb
#
# Copyright 2019 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

IndexType = Union[int, Tensor]
KeyType = Union[int, Tensor]
ValueType = Union[NestedTensor, Sequence[NestedTensor]]


class ReplayBuffer(remote.Remotable, Launchable):

    def __init__(self,
                 storage: Storage,
                 sampler: Sampler,
                 identifier: Optional[str] = None) -> None:
        remote.Remotable.__init__(self, identifier)

        self._storage = storage
        self._sampler = sampler

    def __len__(self) -> int:
        return len(self._storage)

    def __getitem__(self, index: IndexType) -> Tuple[KeyType, ValueType]:
        return self._storage.at(index)

    @property
    def capacity(self) -> int:
        return self._storage.capacity

    @property
    def size(self) -> int:
        return self._storage.size

    def init_launching(self) -> None:
        pass

    def init_execution(self) -> None:
        pass

    @remote.remote_method(batch_size=None)
    def info(self) -> Tuple[int, int]:
        return self.size, self.capacity

    @remote.remote_method(batch_size=None)
    def reset(self) -> None:
        self._storage.reset()
        self._sampler.reset()

    @remote.remote_method(batch_size=None)
    def clear(self) -> None:
        self._storage.clear()
        self._sampler.reset()

    @remote.remote_method(batch_size=None)
    def at(self, index: IndexType) -> Tuple[KeyType, ValueType]:
        return self._storage.at(index)

    @remote.remote_method(batch_size=None)
    def get(self, key: KeyType) -> ValueType:
        return self._storage.get(key)

    @remote.remote_method(batch_size=None)
    def append(self, data: NestedTensor, priority: float = 1.0) -> int:
        new_key, old_key = self._storage.append(data)
        self._sampler.insert(new_key, priority)
        if old_key is not None:
            self._sampler.delete(old_key)
        return new_key

    @remote.remote_method(batch_size=None)
    def extend(self,
               data: Sequence[NestedTensor],
               priorities: Union[float, Tensor] = 1.0) -> torch.Tensor:
        new_keys, old_keys = self._storage.extend(data)
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.numpy().astype(np.float64)
        elif isinstance(priorities, np.ndarray):
            priorities = priorities.astype(np.float64)
        self._sampler.insert(new_keys, priorities)
        self._sampler.delete(old_keys)
        return torch.from_numpy(new_keys)

    @remote.remote_method(batch_size=None)
    def sample(
        self,
        num_samples: int,
        replacement: bool = False
    ) -> Tuple[torch.Tensor, NestedTensor, torch.Tensor]:
        keys, probabilities = self._sampler.sample(num_samples, replacement)
        values = self._storage.get(keys)
        return torch.from_numpy(keys), values, torch.from_numpy(probabilities)

    @remote.remote_method(batch_size=None)
    def update(self, key: Union[int, Tensor], priority: Union[float,
                                                              Tensor]) -> None:
        if isinstance(key, torch.Tensor):
            key = key.numpy()
        if isinstance(priority, torch.Tensor):
            priority = priority.numpy().astype(np.float64)
        elif isinstance(priority, np.ndarray):
            priority = priority.astype(np.float64)
        self._sampler.update(key, priority)


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
        self,
        num_samples: int,
        replacement: bool = False
    ) -> Union[NestedTensor, Tuple[NestedTensor, torch.Tensor, torch.Tensor,
                                   torch.Tensor]]:
        if len(self._futures) > 0:
            ret = self._futures.popleft().result()
        else:
            ret = self.client.sync(self.server_name,
                                   self.remote_method_name("sample"),
                                   num_samples, replacement)

        while len(self._futures) < self.prefetch:
            fut = self.client.async_(self.server_name,
                                     self.remote_method_name("sample"),
                                     num_samples, replacement)
            self._futures.append(fut)

        return ret

    async def async_sample(
        self,
        num_samples: int,
        replacement: bool = False
    ) -> Union[NestedTensor, Tuple[NestedTensor, torch.Tensor, torch.Tensor,
                                   torch.Tensor]]:
        if len(self._futures) > 0:
            ret = await self._futures.popleft()
        else:
            ret = await self.client.async_(self.server_name,
                                           self.remote_method_name("sample"),
                                           num_samples, replacement)

        while len(self._futures) < self.prefetch:
            fut = self.client.async_(self.server_name,
                                     self.remote_method_name("sample"),
                                     num_samples, replacement)
            self._futures.append(fut)

        return ret

    def warm_up(self, learning_starts: Optional[int] = None) -> None:
        size, capacity = self.info()
        target_size = capacity
        if learning_starts is not None:
            target_size = min(target_size, learning_starts)
        width = len(str(capacity)) + 1
        while size < target_size:
            time.sleep(1)
            size, capacity = self.info()
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
