from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List

from pose_providers.mediapipe_pose import KP_INDEX


@dataclass
class Event:
    t: float
    name: str


class ArmRaiseFSM:
    """Trigger when either wrist stays above its shoulder for N frames."""

    def __init__(self, margin: float = 0.05, min_frames: int = 3, cooldown_ms: int = 200):
        self.margin = margin
        self.min_frames = min_frames
        self.cooldown = cooldown_ms / 1000.0
        self.count_l = 0
        self.count_r = 0
        self.last_t = -np.inf

    def step(self, xy: np.ndarray, t: float) -> Event | None:
        k = KP_INDEX
        cond_l = xy[k['l_wrist'], 1] < xy[k['l_shoulder'], 1] - self.margin
        cond_r = xy[k['r_wrist'], 1] < xy[k['r_shoulder'], 1] - self.margin
        self.count_l = self.count_l + 1 if cond_l else 0
        self.count_r = self.count_r + 1 if cond_r else 0
        if (self.count_l >= self.min_frames or self.count_r >= self.min_frames) and (t - self.last_t) > self.cooldown:
            self.last_t = t
            self.count_l = self.count_r = 0
            return Event(t, 'arm_raise')
        return None


class SquatFSM:
    """Counts a squat on the rising edge."""

    def __init__(self, cooldown_ms: int = 200):
        self.cooldown = cooldown_ms / 1000.0
        self.down = False
        self.last_t = -np.inf

    def step(self, knee_min: float, hip_drop: float, t: float) -> Event | None:
        down_cond = (knee_min < 100.0) and (hip_drop > 0.06)
        up_cond = (knee_min > 150.0) or (hip_drop < 0.02)
        if not self.down and down_cond:
            self.down = True
        elif self.down and up_cond:
            self.down = False
            if (t - self.last_t) > self.cooldown:
                self.last_t = t
                return Event(t, 'squat')
        return None


class SitDownFSM:
    """Detect sitting when posture stable for M frames."""

    def __init__(self, M: int = 12, cooldown_ms: int = 200):
        self.M = M
        self.cooldown = cooldown_ms / 1000.0
        self.counter = 0
        self.in_pose = False
        self.last_t = -np.inf

    def step(self, knee_l: float, knee_r: float, hip_ys: List[float], hip_drop: float, t: float) -> Event | None:
        if len(hip_ys) < self.M:
            self.counter = 0
            return None
        var = float(np.var(hip_ys))
        cond = (70.0 <= knee_l <= 110.0 and 70.0 <= knee_r <= 110.0 and var < 0.004 and hip_drop > 0.03)
        if cond:
            self.counter += 1
            if not self.in_pose and self.counter >= self.M and (t - self.last_t) > self.cooldown:
                self.in_pose = True
                self.last_t = t
                return Event(t, 'sit_down')
        else:
            self.counter = 0
            self.in_pose = False
        return None
