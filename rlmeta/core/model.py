# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import functools

from enum import IntEnum
from typing import (Any, Awaitable, Callable, Dict, Optional, Sequence, Tuple,
                    Union)

import numpy as np

import torch
import torch.nn as nn

import rlmeta.core.remote as remote
import rlmeta.ops as rlmeta_ops
import rlmeta.utils.data_utils as data_utils
import rlmeta.utils.nested_utils as nested_utils

from rlmeta.core.launchable import Launchable
from rlmeta.core.server import Server
from rlmeta.core.types import NestedTensor
from rlmeta.samplers import UniformSampler
from rlmeta.storage.circular_buffer import CircularBuffer


class ModelVersion(IntEnum):
    # Use negative values for latest version flag to avoid conflict with real
    # version.
    LATEST = -0x7FFFFFFF
    STABLE = -1


class RemotableModel(nn.Module, remote.Remotable):

    def __init__(self, identifier: Optional[str] = None) -> None:
        nn.Module.__init__(self)
        remote.Remotable.__init__(self, identifier)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


class RemotableModelPool(remote.Remotable, Launchable):

    def __init__(self,
                 model: RemotableModel,
                 capacity: int = 0,
                 identifier: Optional[str] = None) -> None:
        super().__init__(identifier)

        self._model = model
        self._capacity = capacity

        if self._capacity > 0:
            self._history = CircularBuffer(self._capacity)

    @property
    def capacity(self) -> int:
        return self._capacity

    def init_launching(self) -> None:
        self._model.share_memory()

    def init_execution(self) -> None:
        self._bind()

    def model(self, version: int = ModelVersion.LATEST) -> nn.Module:
        return self._model if version == ModelVersion.LATEST else self._history[
            version][1]

    @remote.remote_method(batch_size=None)
    def pull(self,
             version: int = ModelVersion.LATEST) -> Dict[str, torch.Tensor]:
        state_dict = self.model(version).state_dict()
        state_dict = nested_utils.map_nested(lambda x: x.cpu(), state_dict)
        return state_dict

    @remote.remote_method(batch_size=None)
    def push(self, state_dict: Dict[str, torch.Tensor]) -> None:
        # Move state_dict to device before loading.
        # https://github.com/pytorch/pytorch/issues/34880
        device = self._model.device
        state_dict = nested_utils.map_nested(lambda x: x.to(device), state_dict)
        self._model.load_state_dict(state_dict)

    @remote.remote_method(batch_size=None)
    def release(self) -> None:
        if self._capacity > 0:
            self._history.append(copy.deepcopy(self._model))

    @remote.remote_method(batch_size=None)
    def sample_model(self) -> int:
        if self._capacity == 0:
            return ModelVersion.LATEST
        else:
            return np.random.randint(len(self._history))

    def _bind(self) -> None:
        for method in self._model.remote_methods:
            batch_size = getattr(getattr(self._model, method), "__batch_size__",
                                 None)
            method_name, method_impl = self._wrap_remote_method(
                method, batch_size)
            self.__remote_methods__.append(method_name)
            setattr(self, method_name, method_impl)
            for i in range(self._capacity):
                method_name, method_impl = self._wrap_remote_method(
                    method, batch_size, i)
                self.__remote_methods__.append(method_name)
                setattr(self, method_name, method_impl)
                method_name, method_impl = self._wrap_remote_method(
                    method, batch_size, -i - 1)
                setattr(self, method_name, method_impl)
                self.__remote_methods__.append(method_name)

    def _wrap_remote_method(
            self,
            method: str,
            batch_size: Optional[int] = None,
            version: int = ModelVersion.LATEST) -> Callable[..., Any]:

        method_name = method
        if version != ModelVersion.LATEST:
            method_name += f"[{version}]"

        method_impl = functools.partial(self._dispatch_model_call, version,
                                        method)

        setattr(method_impl, "__remote__", True)
        if batch_size is not None:
            setattr(method_impl, "__batch_size__", batch_size)

        return method_name, method_impl

    def _dispatch_model_call(self, version: int, method: str, *args,
                             **kwargs) -> Any:
        model = self.model(version)
        device = model.device
        args = nested_utils.map_nested(lambda x: x.to(device), args)
        kwargs = nested_utils.map_nested(lambda x: x.to(device), kwargs)
        ret = getattr(model, method)(*args, **kwargs)
        ret = nested_utils.map_nested(lambda x: x.cpu(), ret)
        return ret


class RemoteModel(remote.Remote):

    def __init__(self,
                 target: RemotableModel,
                 server_name: str,
                 server_addr: str,
                 name: Optional[str] = None,
                 version: int = ModelVersion.LATEST,
                 timeout: float = 60) -> None:
        super().__init__(target, server_name, server_addr, name, timeout)
        self._version = version

    @property
    def version(self) -> int:
        return self._version

    @version.setter
    def version(self, version: int) -> None:
        self._version = version

    def sample_model(self,
                     num_samples: int = 1,
                     replacement: bool = False) -> torch.Tensor:
        return self.client.sync(self.server_name,
                                self.remote_method_name("sample_model"),
                                num_samples, replacement)

    async def async_sample_model(self,
                                 num_samples: int = 1,
                                 replacement: bool = False) -> torch.Tensor:
        return await self.client.async_(self.server_name,
                                        self.remote_method_name("sample_model"),
                                        num_samples, replacement)

    def _bind(self) -> None:
        for method in self._remote_methods:
            method_name = self.remote_method_name(method)
            self._client_methods[method] = functools.partial(
                self._remote_model_call, method_name)
            self._client_methods["async_" + method] = functools.partial(
                self._async_remote_model_call, method_name)

    def _remote_model_call(self, method: str, *args, **kwargs) -> Any:
        method_name = method
        if self._version != ModelVersion.LATEST:
            method_name += f"[{self._version}]"
        return self.client.sync(self.server_name, method_name, *args, **kwargs)

    def _async_remote_model_call(self, method: str, *args,
                                 **kwargs) -> Awaitable:
        method_name = method
        if self._version != ModelVersion.LATEST:
            method_name += f"[{self._version}]"
        return self.client.async_(self.server_name, method_name, *args,
                                  **kwargs)


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

    def pull(self, version: int = ModelVersion.LATEST) -> None:
        state_dict = self.client.sync(self.server_name,
                                      self.remote_method_name("pull"), version)
        self.wrapped.load_state_dict(state_dict)

    async def async_pull(self, version: int = ModelVersion.LATEST) -> None:
        state_dict = await self.client.async_(self.server_name,
                                              self.remote_method_name("pull"),
                                              version)
        self.wrapped.load_state_dict(state_dict)

    def push(self) -> None:
        state_dict = self.wrapped.state_dict()
        state_dict = nested_utils.map_nested(lambda x: x.cpu(), state_dict)
        self.client.sync(self.server_name, self.remote_method_name("push"),
                         state_dict)

    async def async_push(self) -> None:
        state_dict = self.wrapped.state_dict()
        state_dict = nested_utils.map_nested(lambda x: x.cpu(), state_dict)
        await self.client.async_(self.server_name,
                                 self.remote_method_name("push"), state_dict)

    def release(self) -> None:
        self.client.sync(self.server_name, self.remote_method_name("release"))

    async def async_release(self) -> None:
        await self.client.async_(self.server_name,
                                 self.remote_method_name("release"))

    def sample_model(self,
                     num_samples: int = 1,
                     replacement: bool = False) -> torch.Tensor:
        return self.client.sync(self.server_name,
                                self.remote_method_name("sample_model"),
                                num_samples, replacement)

    async def async_sample_model(self,
                                 num_samples: int = 1,
                                 replacement: bool = False) -> torch.Tensor:
        return await self.client.async_(self.server_name,
                                        self.remote_method_name("sample_model"),
                                        num_samples, replacement)

    def _bind(self) -> None:
        pass


ModelLike = Union[nn.Module, RemotableModel, RemoteModel, DownstreamModel,
                  remote.Remote]


def make_remote_model(model: RemotableModel,
                      server: Server,
                      name: Optional[str] = None,
                      version: int = ModelVersion.LATEST,
                      timeout: float = 60) -> RemoteModel:
    return RemoteModel(model, server.name, server.addr, name, version, timeout)


def wrap_downstream_model(model: RemotableModel,
                          server: Server,
                          name: Optional[str] = None,
                          timeout: float = 60) -> DownstreamModel:
    return DownstreamModel(model, server.name, server.addr, name, timeout)
