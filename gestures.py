from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from pose_providers.mediapipe_pose import KP_INDEX

@dataclass
class Event:
    t: float
    name: str

class ArmRaiseFSM:
    """Dispara quando um pulso fica acima do ombro por N frames."""
    def __init__(self, margin: float = 0.05, min_frames: int = 3, cooldown_ms: int = 200):
        self.margin = margin
        self.min_frames = min_frames
        self.cooldown = cooldown_ms / 1000.0
        self.count_l = 0
        self.count_r = 0
        self.last_t = -np.inf

    def step(self, xy: np.ndarray, t: float) -> Optional[Event]:
        k = KP_INDEX
        cond_l = xy[k["left_wrist"], 1]  < xy[k["left_shoulder"], 1]  - self.margin
        cond_r = xy[k["right_wrist"], 1] < xy[k["right_shoulder"], 1] - self.margin
        self.count_l = self.count_l + 1 if cond_l else 0
        self.count_r = self.count_r + 1 if cond_r else 0
        if (self.count_l >= self.min_frames or self.count_r >= self.min_frames) and (t - self.last_t) > self.cooldown:
            self.last_t = t
            self.count_l = self.count_r = 0
            return Event(t, "arm_raise")
        return None

class SquatFSM:
    """
    Conta 1 squat na subida.
    Histerese:
      - descer: knee_min < knee_down  e hip_drop > hip_down
      - subir:  knee_min > knee_up    ou hip_drop < hip_up
    """
    def __init__(self, mode: str = "real", cooldown_ms: int = 250):
        self.cooldown = cooldown_ms / 1000.0
        self.down = False
        self.last_t = -np.inf
        if mode == "cartoon":
            self.knee_down, self.hip_down = 115.0, 0.02
            self.knee_up,   self.hip_up   = 150.0, 0.01
        else:  # real
            self.knee_down, self.hip_down = 110.0, 0.04
            self.knee_up,   self.hip_up   = 150.0, 0.02

    def step(self, knee_min: float, hip_drop: float, t: float) -> Optional[Event]:
        down_cond = (knee_min < self.knee_down) and (hip_drop > self.hip_down)
        up_cond   = (knee_min > self.knee_up) or (hip_drop < self.hip_up)
        if not self.down and down_cond:
            self.down = True
        elif self.down and up_cond:
            self.down = False
            if (t - self.last_t) > self.cooldown:
                self.last_t = t
                return Event(t, "squat")
        return None

class SitDownFSM:
    """
    Detecta sentar quando:
      - joelhos ~90° (com tolerância) nos dois lados
      - quadril rebaixado de forma consistente
      - pouca variação vertical do quadril por M frames
    """
    def __init__(self, mode: str = "real", M: int = 12, cooldown_ms: int = 400):
        self.M = M
        self.cooldown = cooldown_ms / 1000.0
        self.counter = 0
        self.in_pose = False
        self.last_t = -np.inf
        if mode == "cartoon":
            self.knee_lo, self.knee_hi = 70.0, 115.0
            self.min_hip_drop = 0.03
            self.var_thr = 0.006
        else:
            self.knee_lo, self.knee_hi = 75.0, 110.0
            self.min_hip_drop = 0.04
            self.var_thr = 0.004

    def step(self, knee_l: float, knee_r: float, hip_ys: List[float], hip_drop: float, t: float) -> Optional[Event]:
        if len(hip_ys) < self.M:
            self.counter = 0
            return None
        var = float(np.var(hip_ys))
        knees_ok = (self.knee_lo <= knee_l <= self.knee_hi) and (self.knee_lo <= knee_r <= self.knee_hi)
        cond = knees_ok and (hip_drop > self.min_hip_drop) and (var < self.var_thr)
        if cond:
            self.counter += 1
            if not self.in_pose and self.counter >= self.M and (t - self.last_t) > self.cooldown:
                self.in_pose = True
                self.last_t = t
                return Event(t, "sit_down")
        else:
            self.counter = 0
            self.in_pose = False
        return None
