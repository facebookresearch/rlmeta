# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import re

from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from omegaconf import DictConfig, OmegaConf

TIME_FMT = "%Y-%m-%d %H:%M:%S"
RE_PREFIX = "E Epoch [0-9]+ .* "


def parse_config(line: str) -> DictConfig:
    re_cfg = re.compile("{.+}")
    cfg = OmegaConf.create(re_cfg.search(line).group())
    return cfg


def parse_value(line: str) -> float:
    tokens = [x for x in line.split(" ") if len(x) > 0]
    return float(tokens[4])


def plot(log_file: str,
         xkey: str,
         ykey: str,
         fig_file: Optional[str] = None) -> None:
    re_time = re.compile("(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})")
    re_value = re.compile(RE_PREFIX + ykey)
    x = []
    y = []
    t = []
    with open(log_file, "r") as f:
        s = f.readline()
        t0 = datetime.strptime(re_time.search(s).group(), TIME_FMT)
        cfg = parse_config(s)
        index = 0
        for line in f:
            t_match = re_time.search(line)
            if t_match is not None:
                cur_time = datetime.strptime(t_match.group(), TIME_FMT)
            if not re_value.match(line):
                continue
            index += 1
            x.append(index)
            y.append(parse_value(line))
            t.append((cur_time - t0).seconds)

    x = np.array(x)
    y = np.array(y)
    if xkey == "step":
        x = x * cfg.steps_per_epoch
    elif xkey == "time":
        x = np.array(t) / 3600.0
        xkey = xkey + " (hours)"

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
    plot(flags.log_file, flags.xkey, flags.ykey, flags.fig_file)


if __name__ == "__main__":
    main()
