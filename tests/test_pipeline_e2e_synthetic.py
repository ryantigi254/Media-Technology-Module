from __future__ import annotations

from pathlib import Path

import pytest

_ = pytest.importorskip("cv2")

from ssc.config import MotionConfig, RecorderConfig
from ssc.pipeline import process_video


def test_pipeline_end_to_end_on_synthetic_video(tmp_path: Path) -> None:
    from tests.conftest import write_synthetic_video

    video = tmp_path / "synthetic.avi"
    write_synthetic_video(video, total_frames=40, move_start=10)

    out_dir = tmp_path / "outputs"

    motion_cfg = MotionConfig(
        min_contour_area=50,
        blur_ksize=1,
        mog2_history=20,
        mog2_detect_shadows=False,
        threshold_value=127,
    )
    recorder_cfg = RecorderConfig(post_event_frames=5)

    code = process_video(
        input_path=video,
        output_dir=out_dir,
        motion_cfg=motion_cfg,
        recorder_cfg=recorder_cfg,
        display=False,
        show_mask=False,
        max_frames=40,
    )
    assert code == 0

    clips = list(out_dir.rglob("*.mp4")) + list(out_dir.rglob("*.avi"))
    metas = list(out_dir.rglob("*.json"))

    assert len(clips) >= 1
    assert len(metas) >= 1
