from __future__ import annotations

import sys
from pathlib import Path

import pytest


root = Path(__file__).resolve().parents[1]
src = root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))


def write_synthetic_video(
    path: Path,
    *,
    width: int = 160,
    height: int = 120,
    fps: int = 10,
    total_frames: int = 40,
    move_start: int = 10,
) -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    path.parent.mkdir(parents=True, exist_ok=True)

    # AVI + MJPG tends to be the most portable option in OpenCV on Windows.
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter for synthetic test video")

    try:
        for i in range(total_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)

            # simple moving block so motion is guaranteed
            if i >= move_start:
                x = 10 + (i - move_start) * 3
                x = int(min(max(x, 0), width - 30))
                cv2.rectangle(frame, (x, 40), (x + 20, 70), (255, 255, 255), -1)

            writer.write(frame)
    finally:
        writer.release()
