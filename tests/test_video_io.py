from __future__ import annotations

from pathlib import Path

import pytest

_ = pytest.importorskip("cv2")

from ssc.video_io import get_video_info, open_capture


def test_open_capture_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "nope.avi"
    with pytest.raises(FileNotFoundError):
        _ = open_capture(missing)


def test_open_capture_and_get_info(tmp_path: Path) -> None:
    from tests.conftest import write_synthetic_video

    vid = tmp_path / "synthetic.avi"
    write_synthetic_video(vid, width=160, height=120, fps=10, total_frames=15)

    cap = open_capture(vid)
    try:
        info = get_video_info(cap)
        assert info.width == 160
        assert info.height == 120
        assert info.fps > 0
    finally:
        cap.release()
