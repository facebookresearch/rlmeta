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
from rlmeta.data import SumSegmentTree, MinSegmentTree
from rlmeta.data import TimestampManager

console = Console()


class ReplayBuffer(remote.Remotable, Launchable):

    def __init__(self,
                 capacity: int,
                 collate_fn: Optional[Callable[[Sequence[NestedTensor]],
                                               NestedTensor]] = None,
                 identifier: Optional[str] = None):
        remote.Remotable.__init__(self, identifier)

        self._buffer = CircularBuffer(capacity)
        if collate_fn is not None:
            self._collate_fn = collate_fn
        else:
            self._collate_fn = data_utils.stack_tensors

    def __len__(self) -> int:
        return len(self._buffer)

    def __getitem__(self, index: Union[int, Tensor]) -> NestedTensor:
        data = self._buffer[index]
        if not isinstance(index, int):
            data = nested_utils.collate_nested(self._collate_fn, data)
        return data

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
    def get_size(self) -> int:
        return self.size

    @remote.remote_method(batch_size=None)
    def get_capacity(self) -> int:
        return self.capacity

    @remote.remote_method(batch_size=None)
    def append(self, data: NestedTensor) -> None:
        self._append(data)

    @remote.remote_method(batch_size=None)
    def extend(self, data: Sequence[NestedTensor]) -> None:
        self._extend(data)

    @remote.remote_method(batch_size=None)
    def sample(self, batch_size: int) -> NestedTensor:
        index = np.random.randint(0, self.size, size=batch_size)
        return self.__getitem__(index)

    def _append(self, data: NestedTensor) -> int:
        return self._buffer.append(data)

    def _extend(self, data: Sequence[NestedTensor]) -> np.ndarray:
        return self._buffer.extend(data)


class PrioritizedReplayBuffer(ReplayBuffer):

    def __init__(self,
                 capacity: int,
                 alpha: float,
                 beta: float,
                 eps: float = 1e-8,
                 collate_fn: Optional[Callable[[Sequence[NestedTensor]],
                                               NestedTensor]] = None,
                 priority_type: np.dtype = np.float64,
                 identifier: Optional[str] = None) -> None:
        super().__init__(capacity, collate_fn, identifier)

        assert alpha > 0
        assert beta >= 0
        self._alpha = alpha
        self._beta = beta
        self._eps = eps

        self._priority_type = priority_type
        self._sum_tree = SumSegmentTree(capacity, dtype=priority_type)
        self._min_tree = MinSegmentTree(capacity, dtype=priority_type)
        self._max_priority = 1.0

        self._timestamps = TimestampManager(capacity)

    def __getitem__(
        self, index: Union[int, Tensor]
    ) -> Tuple[NestedTensor, Union[float, torch.Tensor]]:
        data = super().__getitem__(index)
        weight = self._compute_weight(index)
        return data, weight

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def beta(self) -> float:
        return self._beta

    @property
    def eps(self) -> float:
        return self._eps

    @property
    def priority_type(self) -> np.dtype:
        return self._priority_type

    @property
    def max_priority(self) -> float:
        return self._max_priority

    @property
    def current_timestamp(self) -> int:
        return self._timestamps.current_timestamp

    @remote.remote_method(batch_size=None)
    def append(self,
               data: NestedTensor,
               priority: Optional[Union[float, Tensor]] = None) -> None:
        self._append(data, priority)

    @remote.remote_method(batch_size=None)
    def extend(self,
               data: Sequence[NestedTensor],
               priority: Optional[Union[float, Tensor]] = None) -> None:
        self._extend(data, priority)

    @remote.remote_method(batch_size=None)
    def sample(
        self, batch_size: int
    ) -> Tuple[NestedTensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        data, weight, index, timestamp = self._sample(batch_size)
        return data, weight, torch.from_numpy(index), torch.from_numpy(
            timestamp)

    @remote.remote_method(batch_size=None)
    def update_priority(self,
                        index: Union[int, Tensor],
                        priority: Union[float, Tensor],
                        timestamp: Optional[Union[int, Tensor]] = None) -> None:
        self._update_priority(index, priority, timestamp)

    def warm_up(self, learning_starts: Optional[int] = None) -> None:
        capacity = self.get_capacity()
        target_size = capacity
        if learning_starts is not None:
            target_size = min(target_size, learning_starts)
        width = len(str(capacity)) + 1
        cur_size = self.get_size()
        while cur_size < target_size:
            time.sleep(1)
            cur_size = self.get_size()
            logging.info("Warming up replay buffer: " +
                         f"[{cur_size: {width}d} / {capacity} ]")

    def _init_priority(self, index) -> None:
        priority = self._max_priority**self.alpha
        self._sum_tree[index] = priority
        self._min_tree[index] = priority

    def _update_priority(
            self,
            index: Union[int, Tensor],
            priority: Union[float, Tensor],
            timestamp: Optional[Union[int, Tensor]] = None) -> None:
        mask = self._timestamps.is_available(
            index, timestamp) if timestamp is not None else None

        priority += self.eps
        if isinstance(priority, float):
            self._max_priority = max(self._max_priority, priority)
        elif mask is None:
            self._max_priority = max(self._max_priority, priority.max().item())
        else:
            self._max_priority = max(self._max_priority,
                                     priority[mask].max().item())
        priority = priority**self.alpha

        if isinstance(priority, np.ndarray):
            priority = priority.astype(self.priority_type)
        elif isinstance(priority, torch.Tensor):
            priority = priority.to(
                data_utils.numpy_dtype_to_torch(self.priority_type))

        if isinstance(index, int):
            if mask is None or mask:
                self._sum_tree[index] = priority
                self._min_tree[index] = priority
        else:
            self._sum_tree.update(index, priority, mask)
            self._min_tree.update(index, priority, mask)

    def _compute_weight(
            self, index: Union[int, Tensor]) -> Union[float, torch.Tensor]:
        p = self._sum_tree[index]
        if isinstance(p, np.ndarray):
            p = torch.from_numpy(p)
        p_min = self._min_tree.query(0, self.capacity)

        # Importance sampling weight formula:
        #   w_i = (p_i / sum(p) * N) ^ (-beta)
        #   weight_i = w_i / max(w)
        #   weight_i = (p_i / sum(p) * N) ^ (-beta) /
        #       ((min(p) / sum(p) * N) ^ (-beta))
        #   weight_i = ((p_i / sum(p) * N) / (min(p) / sum(p) * N)) ^ (-beta)
        #   weight_i = (p_i / min(p)) ^ (-beta)
        return (p / p_min)**(-self.beta)

    def _append(self,
                data: NestedTensor,
                priority: Optional[Union[float, Tensor]] = None) -> int:
        index = super()._append(data)
        if priority is None:
            self._init_priority(index)
        else:
            self._update_priority(index, priority)
        self._update_timestamp(index)
        return index

    def _extend(self,
                data: Sequence[NestedTensor],
                priority: Optional[Union[float, Tensor]] = None) -> np.ndarray:
        index = super()._extend(data)
        if priority is None:
            self._init_priority(index)
        else:
            self._update_priority(index, priority)
        self._timestamps.update(index)
        return index

    def _sample(
        self, batch_size: int
    ) -> Tuple[NestedTensor, torch.Tensor, np.ndarray, np.ndarray]:
        p_sum = self._sum_tree.query(0, self.capacity)
        mass = np.random.uniform(0.0, p_sum,
                                 size=batch_size).astype(self.priority_type)
        index = self._sum_tree.scan_lower_bound(mass)
        data, weight = self.__getitem__(index)
        timestamp = self._timestamps[index]
        return data, weight, index, timestamp


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
        capacity = self.get_capacity()
        target_size = capacity
        if learning_starts is not None:
            target_size = min(target_size, learning_starts)
        width = len(str(capacity)) + 1
        cur_size = self.get_size()
        while cur_size < target_size:
            time.sleep(1)
            cur_size = self.get_size()
            console.log("Warming up replay buffer: " +
                        f"[{cur_size: {width}d} / {capacity} ]")


ReplayBufferLike = Union[ReplayBuffer, RemoteReplayBuffer]


def make_remote_replay_buffer(target: ReplayBuffer,
                              server: Server,
                              name: Optional[str] = None,
                              prefetch: int = 0,
                              timeout: float = 60):
    return RemoteReplayBuffer(target, server.name, server.addr, name, prefetch,
                              timeout)
