# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import threading

from typing import Any, Callable, Optional, NoReturn

import rlmeta.utils.data_utils as data_utils

import _rlmeta_extension.rpc as rpc
import _rlmeta_extension.rpc.rpc_utils as rpc_utils


class Server(rpc.Server):

    def __init__(self, addr: str):
        super().__init__(addr)
        self._tasks = []
        self._lock = threading.Lock()

    @property
    def addr(self) -> str:
        return super().addr

    def register(self,
                 func_name: str,
                 func_impl: Callable[..., Any],
                 batch_size: Optional[int] = None) -> None:
        if batch_size is None:
            batch_size = 0
        q = self.register_queue(func_name, batch_size)
        t = threading.Thread(target=self._process, args=(q, func_impl))
        self._tasks.append((q, t))

    def start(self) -> None:
        super().start()
        for _, t in self._tasks:
            t.start()

        print("rpc.Server started")

    def stop(self) -> None:
        super().stop()
        for q, t in self._tasks:
            q.shutdown()
            t.join()

        print("rpc.Server stopped")

    def _process(self, queue: rpc.ComputationQueue,
                 func: Callable[..., Any]) -> NoReturn:
        try:
            while True:
                task = queue.get()
                if task is None:
                    break
                self._wrap_func(task, func)
        except StopIteration:
            return

    def _wrap_func(self, task: rpc.Task, func: Callable[..., Any]) -> None:
        batch_size = None
        args = task.args()
        kwargs = task.kwargs()
        if isinstance(task, rpc.BatchedTask):
            batch_size = task.batch_size
            args = data_utils.stack_fields(args)
            kwargs = data_utils.stack_fields(kwargs)

        # Lock to protect any state inside func.
        # TODO: Find a better way to do this (e.g. asyncio)
        with self._lock:
            ret = func(*args, **kwargs)

        if batch_size is not None:
            if ret is None:
                ret = (None,) * batch_size
            else:
                ret = data_utils.unstack_fields(ret, batch_size)

        task.set_return_value(ret)
