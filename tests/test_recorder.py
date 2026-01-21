from __future__ import annotations

import json
from pathlib import Path

import pytest

_ = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from ssc.config import RecorderConfig
from ssc.recorder import IncidentRecorder
from ssc.video_io import VideoInfo


def test_incident_recorder_writes_clip_and_metadata(tmp_path: Path) -> None:
    info = VideoInfo(fps=10.0, width=160, height=120, frame_count=0)
    rec = IncidentRecorder(output_dir=tmp_path, video_info=info, cfg=RecorderConfig(post_event_frames=5))

    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    # trigger recording
    for i in range(3):
        rec.update(i, frame, motion=True)

    # allow it to stop
    for j in range(3, 12):
        rec.update(j, frame, motion=False)

    rec.close(final_frame_idx=12)

    clips = sorted([p for p in tmp_path.iterdir() if p.suffix.lower() in {".mp4", ".avi"}])
    metas = sorted([p for p in tmp_path.iterdir() if p.suffix.lower() == ".json"])

    assert len(clips) == 1
    assert len(metas) == 1

    meta = json.loads(metas[0].read_text(encoding="utf-8"))
    assert meta["frames_written"] > 0
    assert Path(meta["clip_path"]).exists()
