# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from enum import IntFlag
from typing import Dict, Optional, Union

import rlmeta.core.remote as remote

from rlmeta.utils.stats_dict import StatsDict


class Phase(IntFlag):
    NONE = 0
    TRAIN = 1
    EVAL = 2
    BOTH = 3
    TRAIN_ATTACKER = 4
    TRAIN_DETECTOR = 5
    EVAL_ATTACKER = 6
    EVAL_DETECTOR = 7
    


class Controller(remote.Remotable):

    @dataclass
    class PhaseStatus:
        limit: Optional[int] = None
        count: int = 0
        stats: StatsDict = StatsDict()

    def __init__(self, identifier: Optional[str] = None) -> None:
        super().__init__(identifier)
        self._phase = Phase.NONE
        self._status = [
            Controller.PhaseStatus(limit=None, count=0, stats=StatsDict())
            for _ in range(len(Phase))
        ]

    def __repr__(self):
        return f"Controller(phase={self._phase})"

    @remote.remote_method(batch_size=None)
    def reset(self) -> None:
        self._phase = Phase.NONE
        for status in self._status:
            status.limit = None
            status.count = 0
            status.stats.clear()

    @remote.remote_method(batch_size=None)
    def phase(self) -> Phase:
        return self._phase

    @remote.remote_method(batch_size=None)
    def set_phase(self, phase: Phase) -> None:
        self._phase = phase

    @remote.remote_method(batch_size=None)
    def reset_phase(self, phase: Phase, limit: Optional[int] = None) -> None:
        status = self._status[phase]
        status.limit = limit
        status.count = 0
        status.stats.reset()

    @remote.remote_method(batch_size=None)
    def count(self, phase: Phase) -> int:
        return self._status[phase].count

    @remote.remote_method(batch_size=None)
    def stats(self, phase: Phase) -> StatsDict:
        return self._status[phase].stats

    @remote.remote_method(batch_size=None)
    def add_episode(self, phase: Phase, stats: Dict[str, float]) -> None:
        status = self._status[phase]
        if status.limit is None or status.count < status.limit:
            status.count += 1
            status.stats.extend(stats)

class DummyController(Controller):

    def __init__(self, identifier: Optional[str] = None) -> None:
        super().__init__(identifier)
    
    @remote.remote_method(batch_size=None)
    def set_phase(self,
                  phase: Phase,
                  limit: Optional[int] = None,
                  reset: bool = False) -> None:
        pass

ControllerLike = Union[Controller, remote.Remote, DummyController]
