from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from ssc.config import MotionConfig

# References:
# - MOG2 background subtraction: https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html
# - Morphological operations: https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html


@dataclass(frozen=True)
class MotionResult:
    motion: bool
    bboxes: list[tuple[int, int, int, int]]
    motion_mask: np.ndarray


def _merge_bboxes(
    bboxes: list[tuple[int, int, int, int]],
    *,
    padding: int,
) -> list[tuple[int, int, int, int]]:
    if not bboxes:
        return []

    # Expand boxes slightly so nearby blobs (e.g., legs/arms) get merged.
    boxes = []
    for x, y, w, h in bboxes:
        x1 = x - padding
        y1 = y - padding
        x2 = x + w + padding
        y2 = y + h + padding
        boxes.append([x1, y1, x2, y2])

    merged: list[list[int]] = []
    used = [False] * len(boxes)

    for i in range(len(boxes)):
        if used[i]:
            continue

        x1, y1, x2, y2 = boxes[i]
        used[i] = True

        changed = True
        while changed:
            changed = False
            for j in range(len(boxes)):
                if used[j]:
                    continue

                a1, b1, a2, b2 = boxes[j]
                overlap = not (x2 < a1 or a2 < x1 or y2 < b1 or b2 < y1)
                if overlap:
                    x1 = min(x1, a1)
                    y1 = min(y1, b1)
                    x2 = max(x2, a2)
                    y2 = max(y2, b2)
                    used[j] = True
                    changed = True

        merged.append([x1, y1, x2, y2])

    out: list[tuple[int, int, int, int]] = []
    for x1, y1, x2, y2 in merged:
        out.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))

    return out


def _union_bbox(bboxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    if not bboxes:
        return []

    x1 = min(x for x, _, _, _ in bboxes)
    y1 = min(y for _, y, _, _ in bboxes)
    x2 = max(x + w for x, _, w, _ in bboxes)
    y2 = max(y + h for _, y, _, h in bboxes)

    return [(int(x1), int(y1), int(x2 - x1), int(y2 - y1))]


def _largest_bbox(bboxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    if not bboxes:
        return []
    x, y, w, h = max(bboxes, key=lambda b: int(b[2]) * int(b[3]))
    return [(int(x), int(y), int(w), int(h))]


def _find_bboxes_contours(mask: np.ndarray, min_area: int) -> list[tuple[int, int, int, int]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes: list[tuple[int, int, int, int]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # quick filter to ignore tiny flicker/noise (shadows, compression artifacts, etc.)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        bboxes.append((int(x), int(y), int(w), int(h)))

    return bboxes


def _find_bboxes_connected_components(mask: np.ndarray, min_area: int) -> list[tuple[int, int, int, int]]:
    # Connected components can be more stable on some clips, but may be stricter depending
    # on how the mask breaks up.
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    bboxes: list[tuple[int, int, int, int]] = []
    for label_idx in range(1, num_labels):
        x, y, w, h, area = stats[label_idx]
        if int(area) < min_area:
            continue
        bboxes.append((int(x), int(y), int(w), int(h)))

    return bboxes


class MotionDetector:
    def __init__(self, cfg: MotionConfig):
        self.cfg = cfg
        # using MOG2 since it works well for fixed CCTV-style cameras
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=cfg.mog2_history,
            varThreshold=cfg.mog2_var_threshold,
            detectShadows=cfg.mog2_detect_shadows,
        )

        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cfg.morph_kernel, cfg.morph_kernel)
        )

        self._roi_mask: np.ndarray | None = None
        self._roi_shape: tuple[int, int] | None = None

    def _get_roi_mask(self, shape_hw: tuple[int, int]) -> np.ndarray | None:
        if not self.cfg.roi_rects:
            return None

        if self._roi_mask is not None and self._roi_shape == shape_hw:
            return self._roi_mask

        h, w = shape_hw
        mask = np.zeros((h, w), dtype=np.uint8)
        for x1, y1, x2, y2 in self.cfg.roi_rects:
            ix1 = max(int(x1), 0)
            iy1 = max(int(y1), 0)
            ix2 = min(int(x2), w)
            iy2 = min(int(y2), h)
            if ix2 > ix1 and iy2 > iy1:
                mask[iy1:iy2, ix1:ix2] = 255

        self._roi_mask = mask
        self._roi_shape = shape_hw
        return self._roi_mask

    def detect(self, frame_bgr: np.ndarray) -> MotionResult:
        assert frame_bgr is not None and getattr(frame_bgr, "size", 0) > 0
        # background subtraction works best on a static camera
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.cfg.blur_ksize > 1:
            # blur smooths out small pixel noise so the mask is less "speckled"
            gray = cv2.GaussianBlur(gray, (self.cfg.blur_ksize, self.cfg.blur_ksize), 0)

        # learningRate controls how fast the background model updates.
        # Lower values can help stop a moving person being "absorbed" into the background.
        fg = self._subtractor.apply(gray, learningRate=float(self.cfg.mog2_learning_rate))

        # threshold helps get a clean binary mask for contour detection
        _, thresh = cv2.threshold(fg, self.cfg.threshold_value, 255, cv2.THRESH_BINARY)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self._kernel, iterations=1)
        if self.cfg.morph_close_iterations > 0:
            thresh = cv2.morphologyEx(
                thresh, cv2.MORPH_CLOSE, self._kernel, iterations=self.cfg.morph_close_iterations
            )
        # dilation makes blobs a bit more connected (boxes are less jittery)
        if self.cfg.dilate_iterations > 0:
            thresh = cv2.dilate(thresh, self._kernel, iterations=self.cfg.dilate_iterations)

        roi_mask = self._get_roi_mask(thresh.shape[:2])
        if roi_mask is not None:
            thresh = cv2.bitwise_and(thresh, roi_mask)

        if self.cfg.use_connected_components:
            bboxes = _find_bboxes_connected_components(thresh, min_area=self.cfg.min_contour_area)
        else:
            bboxes = _find_bboxes_contours(thresh, min_area=self.cfg.min_contour_area)
        bboxes = _merge_bboxes(bboxes, padding=self.cfg.merge_bbox_padding)
        if self.cfg.single_box:
            bboxes = _largest_bbox(bboxes)

        return MotionResult(motion=len(bboxes) > 0, bboxes=bboxes, motion_mask=thresh)
