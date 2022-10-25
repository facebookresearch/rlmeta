# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np

import rlmeta.utils.nested_utils as nested_utils
import _rlmeta_extension

from rlmeta.core.types import NestedTensor, Tensor
from rlmeta.storage import Storage

IndexType = Union[int, Tensor]
KeyType = Union[int, Tensor]
ValueType = Union[NestedTensor, Sequence[NestedTensor]]


class CircularBuffer(Storage):

    def __init__(
        self,
        capacity: int,
        collate_fn: Optional[Callable[[Sequence[NestedTensor]],
                                      NestedTensor]] = None
    ) -> None:
        self._impl = _rlmeta_extension.CircularBuffer(capacity)
        self._collate_fn = collate_fn

    def __getitem__(self, index: IndexType) -> Tuple[KeyType, ValueType]:
        key, val = self._impl[index]
        if not isinstance(key, int) and self._collate_fn is not None:
            val = nested_utils.collate_nested(self._collate_fn, val)
        return key, val

    @property
    def capacity(self) -> int:
        return self._impl.capacity

    @property
    def size(self) -> int:
        return self._impl.size

    def empty(self) -> bool:
        return self._impl.empty()

    def reset(self) -> None:
        self._impl.reset()

    def clear(self) -> None:
        self._impl.clear()

    def front(self) -> Tuple[KeyType, ValueType]:
        return self._impl.front()

    def back(self) -> Tuple[KeyType, ValueType]:
        return self._impl.back()

    def at(self, index: IndexType) -> Tuple[KeyType, ValueType]:
        key, val = self._impl.at(index)
        if not isinstance(key, int) and self._collate_fn is not None:
            val = nested_utils.collate_nested(self._collate_fn, val)
        return key, val

    def get(self, key: KeyType) -> ValueType:
        val = self._impl.get(key)
        if not isinstance(key, int) and self._collate_fn is not None:
            val = nested_utils.collate_nested(self._collate_fn, val)
        return val

    def append(self, data: NestedTensor) -> Tuple[int, Optional[int]]:
        return self._impl.append(data)

    def extend(self,
               data: Sequence[NestedTensor]) -> Tuple[np.ndarray, np.ndarray]:
        return self._impl.extend(data)
