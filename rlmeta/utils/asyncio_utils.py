# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import asyncio

from typing import Awaitable


def handle_task_exception(task: asyncio.Task) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        pass
    except Exception as e:
        raise e


def create_task(loop: asyncio.BaseEventLoop, coro: Awaitable) -> asyncio.Task:
    task = loop.create_task(coro)
    task.add_done_callback(handle_task_exception)
    return task
