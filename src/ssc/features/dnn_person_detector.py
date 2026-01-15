from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

DNN_PROTO_EXT = ".prototxt"
DNN_MODEL_EXT = ".caffemodel"

# References:
# - MobileNet-SSD (Caffe model files): https://github.com/chuanqi305/MobileNet-SSD
# - OpenCV DNN module: https://docs.opencv.org/4.x/d6/d0f/group__dnn.html


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    return float(inter) / float(union) if union > 0 else 0.0


def _non_max_suppression(boxes: list[tuple[int, int, int, int]], scores: list[float], iou_thresh: float = 0.4) -> list[tuple[int, int, int, int]]:
    # Standard NMS pattern (IoU filtering) like the common OpenCV examples.
    # Reason: this block is typical boilerplate, so it was drafted with assistance and then reviewed.
    if not boxes:
        return []
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    while indices:
        i = indices.pop(0)
        keep.append(boxes[i])
        indices = [j for j in indices if _iou(boxes[i], boxes[j]) < iou_thresh]
    return keep


def _resolve_model_files(model_path: Path) -> tuple[Path, Path] | None:
    if not model_path.is_file():
        return None
    proto = model_path.with_suffix(DNN_PROTO_EXT)
    model = model_path.with_suffix(DNN_MODEL_EXT)
    if not proto.is_file() or not model.is_file():
        return None
    return proto, model


def find_default_dnn_model_path(search_root: Path) -> Path | None:
    if search_root is None or not search_root.exists():
        return None

    prototxt_candidates = sorted(search_root.rglob(f"*{DNN_PROTO_EXT}"), key=lambda p: str(p).lower())
    for proto in prototxt_candidates:
        model = proto.with_suffix(DNN_MODEL_EXT)
        if model.is_file():
            return proto

    model_candidates = sorted(search_root.rglob(f"*{DNN_MODEL_EXT}"), key=lambda p: str(p).lower())
    for model in model_candidates:
        proto = model.with_suffix(DNN_PROTO_EXT)
        if proto.is_file():
            return model

    return None


@lru_cache(maxsize=4)
def _load_net(model_path_str: str) -> cv2.dnn_Net | None:
    model_path = Path(model_path_str)
    files = _resolve_model_files(model_path)
    if files is None:
        return None
    proto, model = files
    try:
        return cv2.dnn.readNetFromCaffe(str(proto), str(model))
    except Exception:
        return None


def detect_people_dnn(
    frame_bgr: np.ndarray,
    model_path: Path,
    conf_thresh: float = 0.5,
    nms_thresh: float = 0.4,
) -> list[tuple[int, int, int, int]]:
    """
    Very lightweight person detector using OpenCV DNN with MobileNet-SSD.
    Returns list of (x, y, w, h) boxes.
    """
    assert 0.0 <= conf_thresh <= 1.0
    assert 0.0 < nms_thresh < 1.0
    files = _resolve_model_files(model_path)
    if files is None:
        return []

    net = _load_net(str(model_path.resolve()))
    if net is None:
        return []

    h, w = frame_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(frame_bgr, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    boxes = []
    scores = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < conf_thresh:
            continue
        class_id = int(detections[0, 0, i, 1])
        if class_id != 15:  # MobileNet-SSD (Caffe) uses VOC-style IDs; 15 corresponds to "person".
            continue
        x0 = int(detections[0, 0, i, 3] * w)
        y0 = int(detections[0, 0, i, 4] * h)
        x1 = int(detections[0, 0, i, 5] * w)
        y1 = int(detections[0, 0, i, 6] * h)
        boxes.append((x0, y0, x1 - x0, y1 - y0))
        scores.append(float(confidence))

    return _non_max_suppression(boxes, scores, iou_thresh=nms_thresh)
