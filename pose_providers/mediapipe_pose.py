from __future__ import annotations

from typing import Optional, Tuple

import mediapipe as mp
import numpy as np

from .base import PoseProvider

# List of the 33 MediaPipe keypoint names for CSV headers etc.
KP_NAMES = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner',
    'right_eye', 'right_eye_outer', 'left_ear', 'right_ear', 'mouth_left',
    'mouth_right', 'left_shoulder', 'right_shoulder', 'left_elbow',
    'right_elbow', 'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
    'left_index', 'right_index', 'left_thumb', 'right_thumb', 'left_hip',
    'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
    'left_heel', 'right_heel', 'left_foot_index', 'right_foot_index'
]

# Convenience index lookup
KP_INDEX = {name: i for i, name in enumerate(KP_NAMES)}


class MediaPipePose(PoseProvider):
    """PoseProvider using MediaPipe Pose with 33 landmarks."""

    def __init__(self,
                 static: bool = False,
                 model_complexity: int = 1,
                 det_conf: float = 0.5,
                 track_conf: float = 0.5) -> None:
        self._static = static
        self._model_complexity = model_complexity
        self._det_conf = det_conf
        self._track_conf = track_conf
        self._pose = None

    def start(self) -> None:
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=self._static,
            model_complexity=self._model_complexity,
            min_detection_confidence=self._det_conf,
            min_tracking_confidence=self._track_conf,
        )

    def infer(self, rgb_frame: np.ndarray) -> Tuple[Optional[Tuple[np.ndarray, np.ndarray]], bool]:
        if self._pose is None:
            raise RuntimeError('Pose provider not started')
        res = self._pose.process(rgb_frame)
        if not res.pose_landmarks:
            return (None, None), False
        xy = np.array([[lmk.x, lmk.y] for lmk in res.pose_landmarks.landmark],
                      dtype=np.float32)
        vis = np.array([lmk.visibility for lmk in res.pose_landmarks.landmark],
                        dtype=np.float32)
        return (xy, vis), True

    def stop(self) -> None:
        if self._pose:
            self._pose.close()
        self._pose = None
