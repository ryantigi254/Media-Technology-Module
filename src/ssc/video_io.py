from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    width: int
    height: int
    frame_count: int


def open_capture(video_path: Path) -> cv2.VideoCapture:
    # OpenCV will return a closed capture if the path/codec is wrong
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    return cap


def get_video_info(cap: cv2.VideoCapture) -> VideoInfo:
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    if fps <= 1e-6:
        # some AVI files don't report FPS properly, so default to something sensible
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    return VideoInfo(fps=fps, width=width, height=height, frame_count=frame_count)
