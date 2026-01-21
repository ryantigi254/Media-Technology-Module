from __future__ import annotations

import json
from pathlib import Path

from tools.wisenet_eval import evaluate_video, load_detections, FrameDetections


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_evaluate_video_basic_metrics(tmp_path: Path) -> None:
    gt_path = tmp_path / "gt.json"
    pred_path = tmp_path / "pred.json"

    gt_payload = {
        "frames": [
            {"frame_index": 0, "bboxes": [[0, 0, 10, 10]]},
            {"frame_index": 1, "bboxes": [[0, 0, 10, 10]]},
            {"frame_index": 2, "bboxes": []},
        ]
    }
    pred_payload = {
        "frames": [
            {"frame_index": 0, "bboxes": [[0, 0, 10, 10]]},
            {"frame_index": 1, "bboxes": []},
            {"frame_index": 2, "bboxes": [[0, 0, 10, 10]]},
        ]
    }

    _write_json(gt_path, gt_payload)
    _write_json(pred_path, pred_payload)

    gt = load_detections(gt_path)
    pred = load_detections(pred_path)

    metrics = evaluate_video(
        video_name="video1",
        gt=gt,
        pred=pred,
        fps=30.0,
        entry_tolerance=0,
        exit_tolerance=0,
    )

    assert metrics.tp == 1
    assert metrics.fn == 1
    assert metrics.fp == 1
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.f1 == 0.5
    assert metrics.mean_iou == 1.0
    assert metrics.iou_50_rate == 1.0


def test_load_detections_accepts_detection_list(tmp_path: Path) -> None:
    gt_path = tmp_path / "gt.json"
    payload = [
        {"frame": 3, "detections": [{"x": 1, "y": 2, "w": 3, "h": 4}]},
        {"frame": 4, "detections": [{"xmin": 0, "ymin": 0, "xmax": 5, "ymax": 5}]},
    ]
    _write_json(gt_path, payload)

    detections = load_detections(gt_path)

    assert detections.by_frame[3] == [(1.0, 2.0, 3.0, 4.0)]
    assert detections.by_frame[4] == [(0.0, 0.0, 5.0, 5.0)]
