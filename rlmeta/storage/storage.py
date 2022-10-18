# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc

from typing import Optional, Sequence, Tuple, Union

import numpy as np

from rlmeta.core.types import Tensor, NestedTensor


class Storage(abc.ABC):

    def __len__(self) -> int:
        return self.size

    @abc.abstractmethod
    def __getitem__(
            self,
            key: Union[int,
                       Tensor]) -> Union[NestedTensor, Sequence[NestedTensor]]:
        """
        """

    @property
    @abc.abstractmethod
    def capacity(self) -> int:
        """
        """

    @property
    @abc.abstractmethod
    def size(self) -> int:
        """
        """

    @abc.abstractmethod
    def reset(self) -> None:
        """
        """

    @abc.abstractmethod
    def clear(self) -> None:
        """
        """

    @abc.abstractmethod
    def append(self, data: NestedTensor) -> Tuple[int, Optional[int]]:
        """
        """

    @abc.abstractmethod
    def extend(self,
               data: Sequence[NestedTensor]) -> Tuple[np.ndarray, np.ndarray]:
        """
        """
