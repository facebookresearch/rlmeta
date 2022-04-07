# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import hydra

from omegaconf import DictConfig, OmegaConf


def config_to_json(cfg: OmegaConf) -> str:
    return json.dumps(OmegaConf.to_container(cfg))
