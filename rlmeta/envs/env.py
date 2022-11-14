# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc

from typing import Optional, Type

from rlmeta.core.types import Action, TimeStep
from rlmeta.core.types import NestedTensor


class Env(abc.ABC):

    @abc.abstractmethod
    def reset(self, *args, seed: Optional[int] = None, **kwargs) -> TimeStep:
        """
        Reset env.
        """

    @abc.abstractmethod
    def step(self, action: Action) -> TimeStep:
        """
        Single env step.
        """

    @abc.abstractmethod
    def close(self) -> None:
        """
        Release resources.
        """


class EnvFactory:

    def __init__(self, cls: Type[Env], *args, **kwargs) -> None:
        self._cls = cls
        self._args = args
        self._kwargs = kwargs

    def __call__(self, index: int) -> Env:
        return self._cls(*self._args, **self._kwargs)
