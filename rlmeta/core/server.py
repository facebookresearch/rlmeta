# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import logging

from typing import Any, Callable, List, NoReturn, Optional, Sequence, Union

import torch
import torch.multiprocessing as mp
from rich.console import Console

import moolib

import rlmeta.utils.asyncio_utils as asyncio_utils

from rlmeta.core.launchable import Launchable
from rlmeta.core.remote import Remotable

console = Console()


class Server(Launchable):

    def __init__(self, name: str, addr: str, timeout: float = 60) -> None:
        self._name = name
        self._addr = addr
        self._timeout = timeout

        self._services = []
        self._process = None
        self._server = None
        self._loop = None
        self._tasks = None

    def __repr__(self):
        return f'Server(name={self._name} addr={self._addr})'

    @property
    def name(self) -> str:
        return self._name

    @property
    def addr(self) -> str:
        return self._addr

    @property
    def timeout(self) -> float:
        return self._timeout

    def add_service(self, service: Union[Remotable,
                                         Sequence[Remotable]]) -> None:
        if isinstance(service, (list, tuple)):
            self._services.extend(service)
        else:
            self._services.append(service)

    def start(self) -> None:
        self.init_launching()
        self._process = mp.Process(target=self.run)
        self._process.start()

    def join(self) -> None:
        self._process.join()

    def terminate(self) -> None:
        if self._process is not None:
            self._process.terminate()

    def run(self) -> NoReturn:
        self.init_execution()
        self._start_services()

    def init_launching(self) -> None:
        for service in self._services:
            if isinstance(service, Launchable):
                service.init_launching()

    def init_execution(self) -> None:
        for service in self._services:
            if isinstance(service, Launchable):
                service.init_execution()

        self._server = moolib.Rpc()
        self._server.set_name(self._name)
        self._server.set_timeout(self._timeout)
        console.log(f"Server={self.name} listening to {self._addr}")
        try:
            self._server.listen(self._addr)
        except:
            console.log(f"ERROR on listen({self._addr}) from: server={self}")
            raise

    def _start_services(self) -> NoReturn:
        self._loop = asyncio.get_event_loop()
        self._tasks = []
        console.log(f"Server={self.name} starting services: {self._services}")
        for service in self._services:
            for method in service.remote_methods:
                method_impl = getattr(service, method)
                batch_size = getattr(method_impl, "__batch_size__", None)
                self._add_server_task(service.remote_method_name(method),
                                      method_impl, batch_size)
        try:
            if not self._loop.is_running():
                self._loop.run_forever()
        except Exception as e:
            logging.error(e)
            raise
        finally:
            for task in self._tasks:
                task.cancel()
            self._loop.stop()
            self._loop.close()
        console.log(f"Server={self.name} services started")

    def _add_server_task(self, func_name: str, func_impl: Callable[..., Any],
                         batch_size: Optional[int]) -> None:
        if batch_size is None:
            que = self._server.define_queue(func_name)
        else:
            que = self._server.define_queue(func_name,
                                            batch_size=batch_size,
                                            dynamic_batching=True)
        task = asyncio_utils.create_task(self._loop,
                                         self._async_process(que, func_impl))
        self._tasks.append(task)

    async def _async_process(self, que: moolib.Queue,
                             func: Callable[..., Any]) -> None:
        try:
            while True:
                ret_cb, args, kwargs = await que
                ret = func(*args, **kwargs)
                ret_cb(ret)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logging.error(e)
            raise e


class ServerList:

    def __init__(self, servers: Optional[Sequence[Server]] = None) -> None:
        self._servers = []
        if servers is not None:
            self._servers.extend(servers)

    def __getitem__(self, index: int) -> Server:
        return self._servers[index]

    @property
    def servers(self) -> List[Server]:
        return self._servers

    def append(self, server: Server) -> None:
        self.servers.append(server)

    def extend(self, servers: Union[ServerList, Sequence[Server]]) -> None:
        if isinstance(servers, ServerList):
            self.servers.extend(servers.servers)
        else:
            self.servers.extend(servers)

    def start(self) -> None:
        for server in self.servers:
            server.start()

    def join(self) -> None:
        for server in self.servers:
            server.join()

    def terminate(self) -> None:
        for server in self.servers:
            server.terminate()


ServerLike = Union[Server, ServerList]
