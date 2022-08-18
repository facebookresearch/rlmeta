# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union

import numpy as np
import torch

from _rlmeta_extension import SumSegmentTreeFp32, SumSegmentTreeFp64
from rlmeta.core.types import Tensor

SumSegmentTreeImpl = Union[SumSegmentTreeFp32, SumSegmentTreeFp64]
Index = Union[int, np.ndarray, torch.Tensor]
Value = Union[float, np.ndarray, torch.Tensor]


class SumSegmentTree:

    def __init__(self, size: int, dtype: np.dtype = np.float64) -> None:
        self._dtype = dtype
        if dtype == np.float32:
            self._impl = SumSegmentTreeFp32(size)
        elif dtype == np.float64:
            self._impl = SumSegmentTreeFp64(size)
        else:
            assert False, "Unsupported data type " + str(dtype)

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def size(self) -> int:
        return self._impl.size

    @property
    def capacity(self) -> int:
        return self._impl.capacity

    def __len__(self) -> int:
        return len(self._impl)

    def __getitem__(self, index: Index) -> Value:
        return self._impl[index]

    def at(self, index: Index) -> Value:
        return self._impl.at(index)

    def __setitem__(self, index: Index, value: Value) -> None:
        self._impl[index] = value

    def update(self,
               index: Index,
               value: Value,
               mask: Optional[Tensor] = None) -> None:
        if mask is None:
            self._impl.update(index, value)
        else:
            self._impl.update(index, value, mask)

    def query(self, l: Index, r: Index) -> Value:
        return self._impl.query(l, r)

    def scan_lower_bound(self, value: Value) -> Index:
        return self._impl.scan_lower_bound(value)
