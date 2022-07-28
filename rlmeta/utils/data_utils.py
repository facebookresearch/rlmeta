# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import io
import os

from typing import Any, Dict, Sequence, Tuple, Union

import numpy as np
import torch

import rlmeta.utils.nested_utils as nested_utils

from rlmeta.core.types import Tensor, NestedTensor

_NUMPY_DTYPE_TO_TORCH_MAP = {
    bool: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}

_TORCH_DTYPE_TO_NUMPY_MAP = {
    torch.bool: bool,
    torch.uint8: np.uint8,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.complex64: np.complex64,
    torch.complex128: np.complex128,
}


def numpy_dtype_to_torch(dtype: np.dtype) -> torch.dtype:
    return _NUMPY_DTYPE_TO_TORCH_MAP[dtype]


def torch_dtype_to_numpy(dtype: torch.dtype) -> np.dtype:
    return _TORCH_DTYPE_TO_NUMPY_MAP[dtype]


def size(data: Tensor) -> Sequence[int]:
    if isinstance(data, np.ndarray):
        return data.shape
    elif isinstance(data, torch.Tensor):
        return data.size()
    return ()


def to_numpy(data: Tensor) -> np.ndarray:
    return data.detach().cpu().numpy() if isinstance(data,
                                                     torch.Tensor) else data


def to_torch(data: Tensor) -> torch.Tensor:
    if isinstance(data, np.generic):
        return torch.tensor(data)
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    return data


def stack_tensors(input: Sequence[Tensor]) -> Tensor:
    size = input[0].size()
    # torch.cat is much faster than torch.stack
    # https://github.com/pytorch/pytorch/issues/22462
    return torch.stack(input) if len(size) == 0 else torch.cat(input).view(
        -1, *size)


def cat_fields(input: Sequence[NestedTensor]) -> NestedTensor:
    assert len(input) > 0
    return nested_utils.collate_nested(lambda x: torch.cat(x), input)


def stack_fields(input: Sequence[NestedTensor]) -> NestedTensor:
    assert len(input) > 0
    return nested_utils.collate_nested(stack_tensors, input)


def unstack_fields(input: NestedTensor,
                   batch_size: int) -> Tuple[NestedTensor, ...]:
    if batch_size == 1:
        return (nested_utils.map_nested(lambda x: x.squeeze(0), input),)
    else:
        return nested_utils.unbatch_nested(lambda x: torch.unbind(x), input,
                                           batch_size)


def serialize_to_bytes(data: Any) -> bytes:
    buffer = io.BytesIO()
    torch.save(data, buffer)
    return buffer.getvalue()


def parse_from_bytes(bytes: bytes) -> Any:
    buffer = io.BytesIO(bytes)
    return torch.load(buffer)
