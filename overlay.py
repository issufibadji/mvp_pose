from __future__ import annotations

import cv2
import numpy as np

from pose_providers.mediapipe_pose import KP_INDEX


def draw_skeleton(frame: np.ndarray, xy: np.ndarray, vis: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """Draw a minimal skeleton using a subset of MediaPipe edges."""
    if xy is None:
        return frame
    h, w = frame.shape[:2]
    def pt(i):
        x, y = xy[i]
        return int(x * w), int(y * h)

    edges = [
        ('l_shoulder', 'r_shoulder'),
        ('l_shoulder', 'l_elbow'), ('l_elbow', 'l_wrist'),
        ('r_shoulder', 'r_elbow'), ('r_elbow', 'r_wrist'),
        ('l_shoulder', 'l_hip'), ('r_shoulder', 'r_hip'), ('l_hip', 'r_hip'),
        ('l_hip', 'l_knee'), ('l_knee', 'l_ankle'),
        ('r_hip', 'r_knee'), ('r_knee', 'r_ankle'),
    ]
    for a, b in edges:
        ia, ib = KP_INDEX[a], KP_INDEX[b]
        if vis[ia] >= thr and vis[ib] >= thr:
            cv2.line(frame, pt(ia), pt(ib), (0, 255, 0), 2)
    return frame


def draw_hud(frame: np.ndarray, counts: dict, pose_ok_pct: float, knee_min: float, hip_drop: float) -> np.ndarray:
    """Draw text HUD with gesture counts and diagnostics."""
    def put(txt, y):
        cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    put(f"Arm raises: {counts.get('arm_raise',0)}", 20)
    put(f"Squats: {counts.get('squat',0)}", 45)
    put(f"Sit downs: {counts.get('sit_down',0)}", 70)
    put(f"Pose OK: {pose_ok_pct:.1f}%", 95)
    put(f"knee_min: {knee_min:.1f}", 120)
    put(f"hip_drop: {hip_drop:.3f}", 145)
    return frame
