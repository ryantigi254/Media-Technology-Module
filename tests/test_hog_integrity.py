from __future__ import annotations

import cv2
import numpy as np


def test_hog_descriptor_dimension_matches_compute() -> None:
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    expected = int(hog.getDescriptorSize())

    # Standard pedestrian window size for HOG people detector.
    win_w, win_h = 64, 128
    frame = np.zeros((win_h, win_w), dtype=np.uint8)
    feats = hog.compute(frame)
    assert feats is not None
    assert int(feats.shape[0]) == expected
