# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import functools

from enum import IntEnum
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

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
    # Use negative values for special version flag to avoid conflict with real
    # version.
    LATEST = -1
    STABLE = -2
    RANDOM = -3


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
                 capacity: int = 1,
                 identifier: Optional[str] = None) -> None:
        super().__init__(identifier)

        self._model = model
        self._capacity = capacity
        self._history = CircularBuffer(self._capacity)
        self._sampler = UniformSampler()

    @property
    def capacity(self) -> int:
        return self._capacity

    def init_launching(self) -> None:
        self._model.share_memory()

    def init_execution(self) -> None:
        self._bind()

    def model(self, version: int = ModelVersion.LATEST) -> nn.Module:
        if version == ModelVersion.LATEST:
            return self._model
        elif version == ModelVersion.STABLE:
            return self._history.back()[1]
        else:
            return self._history.get(version)

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
        new_key, old_key = self._history.append(copy.deepcopy(self._model))
        self._sampler.insert(new_key, 1.0)
        if old_key is not None:
            self._sampler.delete(old_key)

    @remote.remote_method(batch_size=None)
    def sample_model(self,
                     num_samples: int = 1,
                     replacement: bool = False) -> torch.Tensor:
        ret, _ = self._sampler.sample(num_samples, replacement)
        return torch.from_numpy(ret)

    def _bind(self) -> None:
        for method in self._model.remote_methods:
            self.__remote_methods__.append(method)
            batch_size = getattr(getattr(self._model, method), "__batch_size__",
                                 None)
            setattr(self, method, self._wrap_remote_method(method, batch_size))

    def _wrap_remote_method(
            self,
            method: str,
            batch_size: Optional[int] = None) -> Callable[..., Any]:

        def wrapped_method(version: torch.Tensor, *args, **kwargs) -> Any:
            return self._dispatch_method_by_version(version, method, *args,
                                                    **kwargs)

        setattr(wrapped_method, "__remote__", True)
        if batch_size is not None:
            setattr(wrapped_method, "__batch_size__", batch_size)

        return wrapped_method

    def _dispatch_method_by_version(self, version: torch.Tensor, method: str,
                                    *args, **kwargs) -> Any:
        version = version.view(-1)
        batch_size = version.numel()
        random_mask = (version == ModelVersion.RANDOM)
        if random_mask.any():
            random_version, _ = self._sampler.sample(batch_size,
                                                     replacement=True)
            random_version = torch.from_numpy(random_version)
            version = torch.where(random_mask, random_version, version)

        values, groups = rlmeta_ops.groupby(version)
        values = values.tolist()
        device = self._model.device
        args = nested_utils.map_nested(lambda x: x.to(device), args)
        kwargs = nested_utils.map_nested(lambda x: x.to(device), kwargs)

        if len(values) == 1:
            ret = self._call_single_model(values[0], method, *args, **kwargs)
            ret = nested_utils.map_nested(lambda x: x.cpu(), ret)
            return ret

        rets = []
        for v, g in zip(values, groups):
            cur_args = nested_utils.map_nested(
                lambda x: self._index_select_data(x, index=g), args)
            cur_kwargs = nested_utils.map_nested(
                lambda x: self._index_select_data(x, index=g), kwargs)
            ret = self._call_single_model(v, method, *cur_args, **cur_kwargs)
            rets.append(ret)
        index = torch.cat(groups)
        index = torch.argsort(index)
        ret = data_utils.cat_fields(rets)
        ret = nested_utils.map_nested(lambda x: x[index, :].cpu(), ret)

        return ret

    def _call_single_model(self, version: int, method: str, *args,
                           **kwargs) -> Any:
        model = self.model(version)
        return getattr(model, method)(*args, **kwargs)

    def _index_select_data(
            self, data: Union[torch.Tensor, Sequence[NestedTensor]],
            index: torch.Tensor) -> Union[torch.Tensor, Tuple[NestedTensor]]:
        if isinstance(data, torch.Tensor):
            return data[index, :]
        else:
            # For Non-tensor data.
            return tuple([data[i] for i in index.tolist()])


class RemoteModel(remote.Remote):

    def __init__(self,
                 target: RemotableModel,
                 server_name: str,
                 server_addr: str,
                 name: Optional[str] = None,
                 version: int = ModelVersion.LATEST,
                 timeout: float = 60) -> None:
        # Define self._version before self._bind in __init__.
        self._version = torch.tensor([version])
        super().__init__(target, server_name, server_addr, name, timeout)

    @property
    def version(self) -> int:
        return self._version.item()

    @version.setter
    def version(self, version: int) -> None:
        self._version = torch.tensor([version])
        # Bind _remote_methods again for new version.
        self._bind()

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
                self.client.sync, self.server_name, method_name, self._version)
            self._client_methods["async_" + method] = functools.partial(
                self.client.async_, self.server_name, method_name,
                self._version)


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
