from __future__ import annotations

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from ssc.config import MotionConfig
from ssc.motion import MotionDetector


def test_motion_detector_no_motion_after_learning() -> None:
    cfg = MotionConfig(
        min_contour_area=50,
        blur_ksize=1,
        mog2_history=20,
        mog2_detect_shadows=False,
        threshold_value=127,
        morph_close_iterations=0,
        dilate_iterations=0,
        single_box=False,
    )
    det = MotionDetector(cfg)

    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    # let the subtractor settle
    for _ in range(25):
        _ = det.detect(frame)

    res = det.detect(frame)
    assert res.motion is False


def test_motion_detector_detects_moving_object() -> None:
    cfg = MotionConfig(
        min_contour_area=50,
        blur_ksize=1,
        mog2_history=20,
        mog2_detect_shadows=False,
        threshold_value=127,
        morph_close_iterations=0,
        dilate_iterations=0,
        single_box=False,
    )
    det = MotionDetector(cfg)

    base = np.zeros((120, 160, 3), dtype=np.uint8)

    for _ in range(20):
        _ = det.detect(base)

    frame = base.copy()
    cv2.rectangle(frame, (40, 40), (70, 80), (255, 255, 255), -1)

    res = det.detect(frame)
    assert res.motion is True
    assert len(res.bboxes) >= 1
