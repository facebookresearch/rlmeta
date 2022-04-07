# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import re

from datetime import datetime
from typing import Any, Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

JSON_REGEX = re.compile("{.+}")


def parse_json(line: str) -> Optional[Dict[str, Any]]:
    m = JSON_REGEX.search(line)
    return None if m is None else json.loads(m.group())


def get_value(val: Union[float, Dict[str, float]]) -> float:
    return val["mean"] if isinstance(val, dict) else val


def plot(log_file: str,
         phase: str,
         xkey: str,
         ykey: str,
         fig_file: Optional[str] = None) -> None:
    x = []
    y = []
    with open(log_file, "r") as f:
        line = f.readline()
        cfg = parse_json(line)
        for line in f:
            stats = parse_json(line)
            if stats is None:
                continue
            cur_phase = stats.get("phase", None)
            if cur_phase == phase:
                x.append(get_value(stats[xkey]))
                y.append(get_value(stats[ykey]))

    x = np.array(x)
    y = np.array(y)

    plt.plot(x, y)
    plt.xlabel(xkey)
    plt.ylabel(ykey)
    if fig_file is not None:
        plt.savefig(fig_file)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, help="log file to plot")
    parser.add_argument("--phase",
                        default="Eval",
                        type=str,
                        help="phase to plot.")
    parser.add_argument("--xkey",
                        default="epoch",
                        type=str,
                        help="x values to plot.")
    parser.add_argument("--ykey",
                        default="episode_return",
                        type=str,
                        help="y values to plot.")
    parser.add_argument("--fig_file",
                        default=None,
                        type=str,
                        help="figure file to save.")

    flags = parser.parse_intermixed_args()
    plot(flags.log_file, flags.phase, flags.xkey, flags.ykey, flags.fig_file)


if __name__ == "__main__":
    main()
