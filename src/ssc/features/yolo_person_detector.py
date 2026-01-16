from __future__ import annotations

from functools import lru_cache

import numpy as np

# This module was AI-assisted and then reviewed.
# Reason: Ultralytics integration, caching, and type annotations are non-trivial and were scaffolded for speed.

# References:
# - Ultralytics YOLO usage: https://docs.ultralytics.com/
# - YOLOv3 paper (context): https://arxiv.org/abs/1804.02767


def is_yolo_available() -> bool:
    try:
        import ultralytics  # noqa: F401

        return True
    except Exception:
        return False


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    from ultralytics import YOLO

    return YOLO(model_name)


def detect_objects_yolo(
    frame_bgr: np.ndarray,
    *,
    model_name: str = "yolov8n.pt",
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    class_names: tuple[str, ...] = ("person", "chair", "tv"),
    return_labels: bool = False,
) -> list[tuple[int, int, int, int]] | list[tuple[tuple[int, int, int, int], str]]:
    """Detect objects using Ultralytics YOLO.

    Returns list of (x, y, w, h) bounding boxes in pixel coordinates.
    If return_labels is True, returns list of ((x, y, w, h), label) tuples.

    Notes:
    - Class filtering uses the model's `names` mapping. If a requested class
      name is not present (e.g., "door" in COCO), it is ignored.
    - If Ultralytics is not installed, this returns [].
    """

    # This wrapper around Ultralytics is boilerplate; drafted with assistance and reviewed.
    assert 0.0 <= conf_thresh <= 1.0
    assert 0.0 < iou_thresh < 1.0

    if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
        return []

    if not is_yolo_available():
        return []

    model = _load_model(str(model_name))

    # Ultralytics expects RGB for numpy arrays in most examples.
    frame_rgb = frame_bgr[..., ::-1]

    results = model.predict(
        source=frame_rgb,
        verbose=False,
        conf=float(conf_thresh),
        iou=float(iou_thresh),
    )

    if not results:
        return []

    r0 = results[0]
    if r0 is None or getattr(r0, "boxes", None) is None:
        return []

    class_filter: list[int] = []
    names = getattr(r0, "names", None)
    if isinstance(names, dict):
        name_to_id = {str(v).lower(): int(k) for k, v in names.items()}
        for name in class_names:
            key = str(name).strip().lower()
            if key in name_to_id:
                class_filter.append(name_to_id[key])

    ids_arr = None
    if r0.boxes.cls is not None:
        ids_arr = r0.boxes.cls.detach().cpu().numpy().astype(int)

    if class_filter and ids_arr is not None:
        keep_mask = np.isin(ids_arr, np.array(class_filter, dtype=int))
    else:
        keep_mask = None

    boxes_xyxy = r0.boxes.xyxy
    if boxes_xyxy is None:
        return []

    arr = boxes_xyxy.detach().cpu().numpy()
    out: list[tuple[int, int, int, int]] = []
    out_labeled: list[tuple[tuple[int, int, int, int], str]] = []
    for idx, (x1, y1, x2, y2) in enumerate(arr):
        if keep_mask is not None and not bool(keep_mask[idx]):
            continue
        ix1 = int(round(float(x1)))
        iy1 = int(round(float(y1)))
        ix2 = int(round(float(x2)))
        iy2 = int(round(float(y2)))
        w = ix2 - ix1
        h = iy2 - iy1
        if w <= 1 or h <= 1:
            continue
        bbox = (ix1, iy1, int(w), int(h))
        out.append(bbox)

        if return_labels:
            label = "object"
            if isinstance(names, dict) and ids_arr is not None and idx < len(ids_arr):
                label = str(names.get(int(ids_arr[idx]), label))
            out_labeled.append((bbox, label))

    return out_labeled if return_labels else out
