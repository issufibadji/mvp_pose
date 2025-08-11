from __future__ import annotations
import numpy as np
from pose_providers.mediapipe_pose import KP_INDEX

EPS = 1e-6

def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Ângulo ABC em graus (2D)."""
    ab, cb = a - b, c - b
    den = np.linalg.norm(ab) * np.linalg.norm(cb) + EPS
    cos = np.dot(ab, cb) / den
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

def angles(xy: np.ndarray) -> dict:
    """Ângulos úteis (graus): braços (cotovelos) e joelhos."""
    k = KP_INDEX
    res = {
        "arm_l":  _angle(xy[k["left_shoulder"]],  xy[k["left_elbow"]],  xy[k["left_wrist"]]),
        "arm_r":  _angle(xy[k["right_shoulder"]], xy[k["right_elbow"]], xy[k["right_wrist"]]),
        "knee_l": _angle(xy[k["left_hip"]],       xy[k["left_knee"]],   xy[k["left_ankle"]]),
        "knee_r": _angle(xy[k["right_hip"]],      xy[k["right_knee"]],  xy[k["right_ankle"]]),
    }
    return res

def torso_len(xy: np.ndarray) -> float:
    k = KP_INDEX
    l_sh, r_sh = xy[k["left_shoulder"]],  xy[k["right_shoulder"]]
    l_hip, r_hip = xy[k["left_hip"]],      xy[k["right_hip"]]
    shoulder_dist = np.linalg.norm(l_sh - r_sh)
    sh_mid = (l_sh + r_sh) / 2
    hip_mid = (l_hip + r_hip) / 2
    mid_dist = np.linalg.norm(sh_mid - hip_mid)
    return float(max(shoulder_dist, mid_dist) + EPS)

def hip_y(xy: np.ndarray) -> float:
    k = KP_INDEX
    return float((xy[k["left_hip"], 1] + xy[k["right_hip"], 1]) / 2)

def hip_drop_norm(hip_y_val: float, baseline: float, torso: float) -> float:
    return max(0.0, (hip_y_val - baseline) / (torso + EPS))

def ema(prev: float | None, x: float, alpha: float = 0.4) -> float:
    return x if prev is None else alpha * x + (1 - alpha) * prev
