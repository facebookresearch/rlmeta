# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
import functools

from typing import Any, Callable, List, Optional

import moolib

from rlmeta.core.launchable import Launchable
from rlmeta.utils.moolib_utils import generate_random_name


class RemotableMeta(abc.ABCMeta):

    def __new__(cls, name, bases, attrs):
        remote_methods = set(attrs.get("__remote_methods__", []))
        for base in bases:
            remote_methods.update(getattr(base, "__remote_methods__", []))
        for method in attrs.values():
            if getattr(method, "__remote__", False):
                remote_methods.add(method.__name__)
        attrs["__remote_methods__"] = list(remote_methods)
        return super().__new__(cls, name, bases, attrs)


class Remotable(abc.ABC, metaclass=RemotableMeta):

    def __init__(self, identifier: Optional[str] = None):
        self._identifier = identifier

    @property
    def remote_methods(self) -> List[str]:
        return getattr(self, "__remote_methods__", [])

    @property
    def identifier(self) -> Optional[str]:
        return self._identifier

    def remote_method_name(self, method: str) -> str:
        return method if self._identifier is None else (self._identifier +
                                                        "::" + method)


class Remote:

    def __init__(self,
                 target: Remotable,
                 server_name: str,
                 server_addr: str,
                 name: Optional[str] = None,
                 timeout: float = 60) -> None:
        self._target_repr = repr(target)
        self._server_name = server_name
        self._server_addr = server_addr

        self._remote_methods = target.remote_methods
        self._identifier = target.identifier
        self._reset(server_name, server_addr, name, timeout)
        self._client_methods = {}

    # TODO: Find a better way to implement this
    def __getattribute__(self, attr: str) -> Any:
        try:
            return object.__getattribute__(self, attr)
        except AttributeError:
            ret = object.__getattribute__(self, "_client_methods").get(attr)
            if ret is not None:
                return ret
            raise

    def __repr__(self):
        return (f"Remote(target={self._target_repr} " +
                f"server_name={self._server_name} " +
                f"server_addr={self._server_addr})")

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name
        if self._client is not None:
            self._client.set_name(name)

    @property
    def server_name(self) -> str:
        return self._server_name

    @property
    def server_addr(self) -> str:
        return self._server_addr

    @property
    def client(self) -> Optional[moolib.Client]:
        return self._client

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def identifier(self) -> Optional[str]:
        return self._identifier

    def remote_method_name(self, method: str) -> str:
        return method if self._identifier is None else (self._identifier +
                                                        "::" + method)

    def connect(self) -> None:
        if self._connected:
            return
        self._client = moolib.Rpc()
        self._client.set_name(self._name)
        self._client.set_timeout(self._timeout)
        self._client.connect(self._server_addr)
        self._bind()
        self._connected = True

    def _reset(self,
               server_name: str,
               server_addr: str,
               name: Optional[str] = None,
               timeout: float = 60) -> None:
        if name is None:
            name = generate_random_name()
        self._server_name = server_name
        self._server_addr = server_addr
        self._name = name
        self._timeout = timeout
        self._client = None
        self._connected = False

    def _bind(self) -> None:
        for method in self._remote_methods:
            method_name = self.remote_method_name(method)
            self._client_methods[method] = functools.partial(
                self.client.sync, self.server_name, method_name)
            self._client_methods["async_" + method] = functools.partial(
                self.client.async_, self.server_name, method_name)


def remote_method(batch_size: Optional[int] = None) -> Callable[..., Any]:

    def remote_method_impl(func: Callable[..., Any]) -> Callable[..., Any]:
        setattr(func, "__remote__", True)
        setattr(func, "__batch_size__", batch_size)
        return func

    return remote_method_impl
