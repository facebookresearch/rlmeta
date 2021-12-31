# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc


class Launchable(abc.ABC):

    @abc.abstractmethod
    def init_launching(self) -> None:
        """
        """

    @abc.abstractmethod
    def init_execution(self) -> None:
        """
        """
