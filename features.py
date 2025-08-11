from __future__ import annotations

import numpy as np

from pose_providers.mediapipe_pose import KP_INDEX

EPS = 1e-6


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return angle ABC in degrees using only x,y."""
    ab, cb = a - b, c - b
    den = np.linalg.norm(ab) * np.linalg.norm(cb) + EPS
    cos = np.dot(ab, cb) / den
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def angles(xy: np.ndarray) -> dict:
    """Compute arm and knee angles in degrees for left and right."""
    k = KP_INDEX
    res = {
        'arm_l': _angle(xy[k['l_shoulder']], xy[k['l_elbow']], xy[k['l_wrist']]),
        'arm_r': _angle(xy[k['r_shoulder']], xy[k['r_elbow']], xy[k['r_wrist']]),
        'knee_l': _angle(xy[k['l_hip']], xy[k['l_knee']], xy[k['l_ankle']]),
        'knee_r': _angle(xy[k['r_hip']], xy[k['r_knee']], xy[k['r_ankle']]),
    }
    return res


def torso_len(xy: np.ndarray) -> float:
    k = KP_INDEX
    l_sh, r_sh = xy[k['l_shoulder']], xy[k['r_shoulder']]
    l_hip, r_hip = xy[k['l_hip']], xy[k['r_hip']]
    shoulder_dist = np.linalg.norm(l_sh - r_sh)
    sh_mid = (l_sh + r_sh) / 2
    hip_mid = (l_hip + r_hip) / 2
    mid_dist = np.linalg.norm(sh_mid - hip_mid)
    return float(max(shoulder_dist, mid_dist) + EPS)


def hip_y(xy: np.ndarray) -> float:
    k = KP_INDEX
    return float((xy[k['l_hip'], 1] + xy[k['r_hip'], 1]) / 2)


def hip_drop_norm(hip_y_val: float, baseline: float, torso: float) -> float:
    return max(0.0, (hip_y_val - baseline) / (torso + EPS))


def ema(prev: float | None, x: float, alpha: float = 0.4) -> float:
    return x if prev is None else alpha * x + (1 - alpha) * prev
