from __future__ import annotations

import pytest

_ = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from ssc.pipeline import _annotate


def test_annotate_draws_bbox() -> None:
    frame = np.zeros((80, 120, 3), dtype=np.uint8)

    # bbox is (x, y, w, h)
    annotated = _annotate(frame, frame_idx=0, bboxes=[(20, 30, 10, 10)], motion=True)

    # rectangle is green in BGR
    assert annotated[30, 20].tolist() == [0, 255, 0]
