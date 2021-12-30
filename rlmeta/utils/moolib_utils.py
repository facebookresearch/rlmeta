# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import uuid


def generate_random_name() -> str:
    return str(uuid.uuid4())


def expend_name_by_index(name: str, index: int) -> str:
    return name + f"-{index}"
