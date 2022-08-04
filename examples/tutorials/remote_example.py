# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import time

import torch
import torch.multiprocessing as mp

import rlmeta.core.remote as remote
import rlmeta.utils.remote_utils as remote_utils

from rlmeta.core.server import Server


class Adder(remote.Remotable):

    @remote.remote_method()
    def add(self, a, b):
        print(f"[Adder.add] a = {a}")
        print(f"[Adder.add] b = {b}")
        return a + b

    @remote.remote_method(batch_size=10)
    def batch_add(self, a, b):
        print(f"[Adder.batch_add] a = {a}")
        print(f"[Adder.batch_add] b = {b}")

        if not isinstance(a, tuple) and not isinstance(b, tuple):
            return a + b
        else:
            return tuple(sum(x) for x in zip(a, b))


async def run_batch(adder_client, send_tensor=False):
    futs = []
    for i in range(20):
        if send_tensor:
            a = torch.tensor([i])
            b = torch.tensor([i + 1])
        else:
            a = i
            b = i + 1
        fut = adder_client.async_batch_add(a, b)
        futs.append(fut)

    await asyncio.sleep(1.0)

    for i, fut in enumerate(futs):
        if send_tensor:
            a = torch.tensor([i])
            b = torch.tensor([i + 1])
        else:
            a = i
            b = i + 1
        c = await fut
        print(f"{a} + {b} = {c}")


def main():
    adder = Adder()
    adder_server = Server(name="adder_server", addr="127.0.0.1:4411")
    adder_server.add_service(adder)
    adder_client = remote_utils.make_remote(adder, adder_server)

    adder_server.start()
    time.sleep(2)
    adder_client.connect()

    a = 1
    b = 2
    c = adder_client.add(a, b)
    print(f"{a} + {b} = {c}")
    # print("")
    # asyncio.run(run_batch(adder_client, send_tensor=False))
    print("")
    asyncio.run(run_batch(adder_client, send_tensor=True))

    adder_server.terminate()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
