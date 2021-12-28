# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from typing import Tuple, Union


class RunningStats(nn.Module):
    def __init__(self, size: Union[int, Tuple[int]]) -> None:
        super().__init__()
        self._size = (size, ) if isinstance(size, int) else size
        self._m0 = 0
        self.register_buffer("_m1", torch.zeros(self._size))
        self.register_buffer("_m2", torch.zeros(self._size))

    def reset(self) -> None:
        self._m0 = 0
        self._m1.zero_()
        self._m2.zero_()

    def count(self) -> int:
        return self._m0

    def mean(self) -> torch.Tensor:
        return self._m1

    def var(self, ddof: int = 0) -> torch.Tensor:
        return self._m2 / (self._m0 - ddof)

    def std(self, ddof: int = 0) -> torch.Tensor:
        return self.var(ddof).sqrt()

    def rstd(self, ddof: int = 0) -> torch.Tensor:
        return self.var(ddof).rsqrt()

    def update(self, x: torch.Tensor) -> None:
        size = x.size()
        if size == self._size:
            self._m0 += 1
            delta = x - self._m1
            self._m1 += delta / self._m0
            self._m2 += delta * (x - self._m1)
        else:
            assert size[1:] == self._size
            m0 = size[0]
            m2, m1 = torch.var_mean(x, dim=0, unbiased=False)
            n = self._m0 + m0
            c = 0.0 if n == 0 else m0 / n
            delta = m1 - self._m1
            self._m1 += c * delta
            self._m2 += m0 * m2 + delta.square() * (c * self._m0)
            self._m0 = n
