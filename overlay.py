# overlay.py — usa MediaPipe names via KP_INDEX
from __future__ import annotations
import cv2
import numpy as np
from pose_providers.mediapipe_pose import KP_INDEX

# aliases curtos (se quiser usar no futuro)
ALIASES = {
    'l_shoulder':'left_shoulder','r_shoulder':'right_shoulder',
    'l_elbow':'left_elbow','r_elbow':'right_elbow',
    'l_wrist':'left_wrist','r_wrist':'right_wrist',
    'l_hip':'left_hip','r_hip':'right_hip',
    'l_knee':'left_knee','r_knee':'right_knee',
    'l_ankle':'left_ankle','r_ankle':'right_ankle',
}
def KP(n: str) -> int: return KP_INDEX[ALIASES.get(n, n)]

def draw_skeleton(frame: np.ndarray, xy: np.ndarray, vis: np.ndarray, thr: float = 0.5) -> np.ndarray:
    if xy is None: return frame
    h, w = frame.shape[:2]
    def pt(i): 
        x, y = xy[i]; return int(x * w), int(y * h)
    edges = [
        ('l_shoulder', 'r_shoulder'),
        ('l_shoulder', 'l_elbow'), ('l_elbow', 'l_wrist'),
        ('r_shoulder', 'r_elbow'), ('r_elbow', 'r_wrist'),
        ('l_shoulder', 'l_hip'), ('r_shoulder', 'r_hip'), ('l_hip', 'r_hip'),
        ('l_hip', 'l_knee'), ('l_knee', 'l_ankle'),
        ('r_hip', 'r_knee'), ('r_knee', 'r_ankle'),
    ]
    for a, b in edges:
        ia, ib = KP(a), KP(b)
        if vis[ia] >= thr and vis[ib] >= thr:
            cv2.line(frame, pt(ia), pt(ib), (0, 255, 0), 2)
    return frame

LABELS = {
    "en": {
        "arm_raise": "Arm raises",
        "squat":     "Squats",
        "sit_down":  "Sit downs",
        "pose_ok":   "Pose OK",
        "knee_min":  "Min knee angle",
        "hip_drop":  "Hip drop (norm.)",
    },
    "pt": {
        "arm_raise": "Elevacao de braco",
        "squat":     "Agachamentos",
        "sit_down":  "Sentar",
        "pose_ok":   "Pose OK",
        "knee_min":  "Angulo minimo do joelho",
        "hip_drop":  "Queda do quadril (norm.)",
    },
}

def draw_hud(frame: np.ndarray, counts: dict, pose_ok_pct: float, knee_min: float, hip_drop: float, lang: str = "pt") -> np.ndarray:
    L = LABELS.get(lang, LABELS["pt"])
    def put(txt, y):
        cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    put(f"{L['arm_raise']}: {counts.get('arm_raise',0)}", 20)
    put(f"{L['squat']}: {counts.get('squat',0)}", 45)
    put(f"{L['sit_down']}: {counts.get('sit_down',0)}", 70)
    put(f"{L['pose_ok']}: {pose_ok_pct:.1f}%", 95)
    put(f"{L['knee_min']}: {knee_min:.1f}°", 120)
    put(f"{L['hip_drop']}: {hip_drop:.3f}", 145)
    return frame
