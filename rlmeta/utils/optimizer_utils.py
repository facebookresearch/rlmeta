# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Iterable, Dict, Optional, Union

import torch

_NAME_TO_OPTIMIZER = {
    "adadelta": torch.optim.Adadelta,
    "adagrad": torch.optim.Adagrad,
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sparseadam": torch.optim.SparseAdam,
    "adamax": torch.optim.Adamax,
    "asgd": torch.optim.ASGD,
    "lbfgs": torch.optim.LBFGS,
    "nadam": torch.optim.NAdam,
    "radam": torch.optim.RAdam,
    "rmsprop": torch.optim.RMSprop,
    "rprop": torch.optim.Rprop,
    "sgd": torch.optim.SGD,
}


def get_optimizer(
        name: str,
        params: Union[Iterable[torch.Tensor], Dict[str, torch.Tensor]],
        args: Optional[Dict[str, Any]] = None) -> torch.optim.Optimizer:
    optimizer = _NAME_TO_OPTIMIZER[name.lower()]
    return optimizer(params) if args is None else optimizer(params, **args)
