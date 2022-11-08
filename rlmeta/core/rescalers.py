# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from rlmeta.utils.running_stats import RunningMoments, RunningRMS


class Rescaler(nn.Module, abc.ABC):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rescale(x)

    def reset(self) -> None:
        pass

    def update(self, x: torch.Tensor) -> None:
        pass

    @abc.abstractmethod
    def rescale(self, x: torch.Tensor) -> torch.Tensor:
        """
        Do rescale for the input tensor.
        """

    @abc.abstractmethod
    def recover(self, x: torch.Tensor) -> torch.Tensor:
        """
        Undo rescale for the input tensor.
        """


class IdentityRescaler(Rescaler):

    def rescale(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def recover(self, x: torch.Tensor) -> torch.Tensor:
        return x


class RMSRescaler(Rescaler):

    def __init__(self,
                 size: Union[int, Tuple[int]],
                 eps: float = 1e-8,
                 dtype: torch.dtype = torch.float64) -> None:
        super().__init__()
        self._size = size
        self._eps = eps
        self._running_rms = RunningRMS(size, dtype=dtype)

    @property
    def size(self) -> Union[int, Tuple[int]]:
        return self._size

    @property
    def eps(self) -> float:
        return self._eps

    def reset(self) -> None:
        self._running_rms.reset()

    def update(self, x: torch.Tensor) -> None:
        self._running_rms.update(x)

    def rescale(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self._running_rms.rrms(self._eps)).to(x.dtype)

    def recover(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self._running_rms.rms(self._eps)).to(x.dtype)


class MomentsRescaler(Rescaler):

    def __init__(self,
                 size: Union[int, Tuple[int]],
                 ddof: int = 0,
                 eps: float = 1e-8,
                 dtype: torch.dtype = torch.float64) -> None:
        super().__init__()
        self._size = size
        self._ddof = ddof
        self._eps = eps
        self._running_moments = RunningMoments(size, dtype=dtype)

    @property
    def size(self) -> Union[int, Tuple[int]]:
        return self._size

    @property
    def ddof(self) -> int:
        return self._ddof

    @property
    def eps(self) -> float:
        return self._eps

    def reset(self) -> None:
        self._running_moments.reset()

    def update(self, x: torch.Tensor) -> None:
        self._running_moments.update(x)

    def rescale(self, x: torch.Tensor) -> torch.Tensor:
        return x if self._running_moments.count() <= 1 else (
            (x - self._running_moments.mean()) *
            self._running_moments.rstd(self._ddof, self._eps)).to(x.dtype)

    def recover(self, x: torch.Tensor) -> torch.Tensor:
        return x if self._running_moments.count() <= 1 else (
            (x * self._running_moments.std(self._ddof, self._eps)) +
            self._running_moments.mean()).to(x.dtype)


class StdRescaler(Rescaler):

    def __init__(self,
                 size: Union[int, Tuple[int]],
                 ddof: int = 0,
                 eps: float = 1e-8,
                 dtype: torch.dtype = torch.float64) -> None:
        super().__init__()
        self._size = size
        self._ddof = ddof
        self._eps = eps
        self._running_moments = RunningMoments(size, dtype=dtype)

    @property
    def size(self) -> Union[int, Tuple[int]]:
        return self._size

    @property
    def ddof(self) -> int:
        return self._ddof

    @property
    def eps(self) -> float:
        return self._eps

    def reset(self) -> None:
        self._running_moments.reset()

    def update(self, x: torch.Tensor) -> None:
        self._running_moments.update(x)

    def rescale(self, x: torch.Tensor) -> torch.Tensor:
        return x if self._running_moments.count() <= 1 else (
            x * self._running_moments.rstd(self._ddof, self._eps)).to(x.dtype)

    def recover(self, x: torch.Tensor) -> torch.Tensor:
        return x if self._running_moments.count() <= 1 else (
            x * self._running_moments.std(self._ddof, self._eps)).to(x.dtype)


class SqrtRescaler(Rescaler):
    """
    Introduced by R2D2 paper.
    https://openreview.net/pdf?id=r1lyTjAqYX
    """

    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self._eps = eps

    @property
    def eps(self) -> float:
        return self._eps

    def rescale(self, x: torch.Tensor) -> torch.Tensor:
        return x.sign() * ((x.abs() + 1.0).sqrt() - 1.0) + self.eps * x

    def recover(self, x: torch.Tensor) -> torch.Tensor:
        if self._eps == 0.0:
            return x.sign() * (x.square() + 2.0 * x.abs())
        r = ((1.0 + 4.0 * self.eps *
              (x.abs() + 1.0 + self.eps)).sqrt() - 1.0) / (2.0 * self.eps)
        return x.sign() * (r.square() - 1.0)
