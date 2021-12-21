# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy

import torch
import torch.nn as nn

import rlmeta.core.remote as remote
import rlmeta_extension.nested_utils as nested_utils

from typing import Any, Dict, Optional, Union

from rlmeta.core.server import Server


class RemotableModel(nn.Module, remote.Remotable):
    def init_launching(self) -> None:
        self.share_memory()

    @remote.remote_method(batch_size=None)
    def pull(self) -> Dict[str, torch.Tensor]:
        state_dict = self.state_dict()
        state_dict = nested_utils.map_nested(lambda x: x.cpu(), state_dict)
        return state_dict

    @remote.remote_method(batch_size=None)
    def push(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.load_state_dict(state_dict)


class DownstreamModel(remote.Remote):
    def __init__(self,
                 model: nn.Module,
                 server_name: str,
                 server_addr: str,
                 name: Optional[str] = None,
                 timeout: float = 60) -> None:
        self._wrapped = model
        self._reset(server_name, server_addr, name, timeout)

    # TODO: Find a better way to implement this
    def __getattribute__(self, attr: str) -> Any:
        try:
            return object.__getattribute__(self, attr)
        except AttributeError:
            return getattr(object.__getattribute__(self, "_wrapped"), attr)

    @property
    def wrapped(self) -> nn.Module:
        return self._wrapped

    def __call__(self, *args, **kwargs) -> Any:
        return self.wrapped(*args, **kwargs)

    def pull(self) -> None:
        state_dict = self.client.sync(self.server_name, "pull")
        self.wrapped.load_state_dict(state_dict)

    async def async_pull(self) -> None:
        state_dict = await self.client.async_(self.server_name, "pull")
        self.wrapped.load_state_dict(state_dict)

    def push(self) -> None:
        state_dict = self.wrapped.state_dict()
        state_dict = nested_utils.map_nested(lambda x: x.cpu(), state_dict)
        self.client.sync(self.server_name, "push", state_dict)

    async def async_push(self) -> None:
        state_dict = self.wrapped.state_dict()
        state_dict = nested_utils.map_nested(lambda x: x.cpu(), state_dict)
        await self.client.async_(self.server_name, "push", state_dict)

    def _bind(self) -> None:
        pass


ModelLike = Union[nn.Module, RemotableModel, DownstreamModel, remote.Remote]


def wrap_downstream_model(model: nn.Module,
                          server: Server,
                          name: Optional[str] = None,
                          timeout: float = 60) -> DownstreamModel:
    return DownstreamModel(model, server.name, server.addr, name, timeout)
