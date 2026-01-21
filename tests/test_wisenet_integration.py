from __future__ import annotations

import os
from pathlib import Path

import pytest

_ = pytest.importorskip("cv2")

from ssc.config import MotionConfig, RecorderConfig
from ssc.pipeline import process_video


@pytest.mark.integration
def test_wisenet_smoke(tmp_path: Path) -> None:
    # This is a smoke test: it checks we can open and process a real WiseNET AVI.
    # It intentionally doesn't assert that motion MUST be found in the first N frames.

    env_dir = os.environ.get("WISENET_DIR")
    if env_dir:
        root = Path(env_dir)
    else:
        # default matches your local download: <repo_root>/data
        repo_root = Path(__file__).resolve().parents[2]
        root = repo_root / "data"

    if not root.exists():
        pytest.skip(f"WiseNET dataset folder not found: {root}")

    videos = sorted(root.rglob("*.avi"))
    if not videos:
        pytest.skip(f"No .avi files found under: {root}")

    video = videos[0]

    out_dir = tmp_path / "wisenet_outputs"

    code = process_video(
        input_path=video,
        output_dir=out_dir,
        motion_cfg=MotionConfig(min_contour_area=800),
        recorder_cfg=RecorderConfig(post_event_frames=10),
        display=False,
        show_mask=False,
        max_frames=60,
    )

    assert code == 0
    assert out_dir.exists()
