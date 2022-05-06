# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union

from rlmeta.core.remote import Remotable, Remote
from rlmeta.core.server import Server


def make_remote(target: Remotable,
                server: Server,
                name: Optional[str] = None,
                timeout: float = 60):
    return Remote(target, server.name, server.addr, name, timeout)
