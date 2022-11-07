# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


class RunningRMS(nn.Module):

    def __init__(self,
                 size: Union[int, Tuple[int]],
                 dtype: Optional[torch.dtype] = None) -> None:
        super().__init__()
        self._size = (size,) if isinstance(size, int) else size
        self.register_buffer("_count", torch.zeros(1, dtype=torch.int64))
        self.register_buffer("_mean_square", torch.zeros(self._size,
                                                         dtype=dtype))

    def reset(self) -> None:
        self._count.zero_()
        self._mean_square.zero_()

    def count(self) -> torch.Tensor:
        return self._count

    def mean_square(self) -> torch.Tensor:
        return self._mean_square

    def rms(self, eps: Optional[float] = None) -> torch.Tensor:
        return self._mean_square.sqrt() if eps is None else ((
            self._mean_square + eps).sqrt())

    def rrms(self, eps: Optional[float] = None) -> torch.Tensor:
        return self._mean_square.rsqrt() if eps is None else ((
            self._mean_square + eps).rsqrt())

    def update(self, x: torch.Tensor) -> None:
        size = x.size()
        if size == self._size:
            self._count += 1
            self._mean_square += (x.square() - self._mean_square) / self._count
        else:
            assert size[1:] == self._size
            cnt = size[0]
            self._count += cnt
            c = 0.0 if self._count == 0 else cnt / self._count
            delta = x.square().mean(dim=0) - self._mean_square
            self._mean_square += c * delta


class RunningMoments(nn.Module):

    def __init__(self,
                 size: Union[int, Tuple[int]],
                 dtype: Optional[torch.dtype] = None) -> None:
        super().__init__()
        self._size = (size,) if isinstance(size, int) else size
        self.register_buffer("_m0", torch.zeros(1, dtype=torch.int64))
        self.register_buffer("_m1", torch.zeros(self._size, dtype=dtype))
        self.register_buffer("_m2", torch.zeros(self._size, dtype=dtype))

    def reset(self) -> None:
        self._m0.zero_()
        self._m1.zero_()
        self._m2.zero_()

    def count(self) -> torch.Tensor:
        return self._m0

    def mean(self) -> torch.Tensor:
        return self._m1

    def var(self, ddof: int = 0) -> torch.Tensor:
        return self._m2 / (self._m0 - ddof)

    def std(self, ddof: int = 0, eps: Optional[float] = None) -> torch.Tensor:
        return self.var(ddof).sqrt() if eps is None else (self.var(ddof) +
                                                          eps).sqrt()

    def rstd(self, ddof: int = 0, eps: Optional[float] = None) -> torch.Tensor:
        return self.var(ddof).rsqrt() if eps is None else (self.var(ddof) +
                                                           eps).rsqrt()

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
