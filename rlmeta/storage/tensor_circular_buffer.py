# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np

import _rlmeta_extension

from rlmeta.core.types import NestedTensor, Tensor
from rlmeta.storage import Storage


class TensorCircularBuffer(Storage):

    def __init__(self, capacity: int) -> None:
        self._impl = _rlmeta_extension.TensorCircularBuffer(capacity)

    def __getitem__(
            self,
            key: Union[int,
                       Tensor]) -> Union[NestedTensor, Sequence[NestedTensor]]:
        return self._impl[key]

    @property
    def capacity(self) -> int:
        return self._impl.capacity

    @property
    def size(self) -> int:
        return self._impl.size

    def clear(self) -> None:
        self._impl.clear()

    def append(self, data: NestedTensor) -> Tuple[int, Optional[int]]:
        return self._impl.append(data)

    def extend(self,
               data: Sequence[NestedTensor]) -> Tuple[np.ndarray, np.ndarray]:
        return self._impl.extend(data)
