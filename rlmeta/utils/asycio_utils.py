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
