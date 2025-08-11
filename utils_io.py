from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Tuple

import cv2
import imageio
import numpy as np

from pose_providers.mediapipe_pose import KP_NAMES

CODECS = ['avc1', 'mp4v', 'XVID']


def open_video_capture(src: str | int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f'Unable to open source {src}')
    return cap


def read_rgb(cap: cv2.VideoCapture) -> Tuple[bool, np.ndarray]:
    ok, frame = cap.read()
    if not ok:
        return False, None
    return True, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def open_video_writer(path: Path, size: Tuple[int, int], fps: float) -> Tuple[cv2.VideoWriter, str]:
    for cc in CODECS:
        fourcc = cv2.VideoWriter_fourcc(*cc)
        writer = cv2.VideoWriter(str(path), fourcc, fps, size)
        if writer.isOpened():
            return writer, cc
    raise RuntimeError('Could not open VideoWriter with available codecs')


def event_writer(path: Path):
    f = open(path, 'w', newline='', encoding='utf-8')
    w = csv.writer(f)
    w.writerow(['t_sec', 'event'])
    return f, w


def keypoint_writer(path: Path):
    f = open(path, 'w', newline='', encoding='utf-8')
    header = ['t_sec', 'pose_ok']
    for name in KP_NAMES:
        header += [f'{name}.x', f'{name}.y', f'{name}.vis']
    w = csv.writer(f)
    w.writerow(header)
    return f, w
