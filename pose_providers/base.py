from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np


class PoseProvider(ABC):
    """Interface for pose providers returning MediaPipe style keypoints."""

    @abstractmethod
    def start(self) -> None:
        """Initialise any resources. Called before infer."""

    @abstractmethod
    def infer(self, rgb_frame: np.ndarray) -> Tuple[Optional[Tuple[np.ndarray, np.ndarray]], bool]:
        """Return ((xy[33,2], vis[33]), pose_ok).

        xy are normalised coordinates in range [0,1] and vis is the landmark
        visibility reported by the backend. pose_ok indicates whether the
        backend produced a result (before visibility gating)."""

    @abstractmethod
    def stop(self) -> None:
        """Release resources."""
