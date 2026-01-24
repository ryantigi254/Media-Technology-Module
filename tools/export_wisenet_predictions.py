from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from ssc.config import HOG_TUNING_DEFAULTS, MotionConfig
from ssc.features.dnn_person_detector import find_default_dnn_model_path, detect_people_dnn
from ssc.features.probabilistic_presence import ProbabilisticPresenceTracker
from ssc.motion import MotionDetector
from ssc.video_io import open_capture, get_video_info

# These evaluation/export scripts were AI-assisted for scaffolding and then verified.
# Reason: they evaluate the implementation but are not required by the brief.


@dataclass(frozen=True)
class FrameRecord:
    frame_number: int
    detections: list[dict[str, Any]]
    presence_probability: float | None = None


def _load_test_items(split_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    if "test" in payload and isinstance(payload["test"], list):
        return payload["test"]
    if "split" in payload and isinstance(payload["split"], dict):
        test_items = payload["split"].get("test")
        if isinstance(test_items, list):
            return test_items
    raise ValueError("Split JSON does not contain a test list.")


def _get_video_path(item: dict[str, Any]) -> Path:
    if "video" in item:
        return Path(item["video"])
    if "video_path" in item:
        return Path(item["video_path"])
    raise ValueError("Split item missing video path.")


def _iter_videos(items: Iterable[dict[str, Any]]) -> Iterable[Path]:
    for item in items:
        yield _get_video_path(item)


def _union_xywh(bboxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
    if not bboxes:
        return None
    xs = [x for x, _, _, _ in bboxes]
    ys = [y for _, y, _, _ in bboxes]
    x2s = [x + w for x, _, w, _ in bboxes]
    y2s = [y + h for _, y, _, h in bboxes]
    x1 = int(min(xs))
    y1 = int(min(ys))
    x2 = int(max(x2s))
    y2 = int(max(y2s))
    return (x1, y1, x2 - x1, y2 - y1)


def _hog_people(frame_bgr: np.ndarray, width: int, height: int, hog: cv2.HOGDescriptor) -> list[tuple[int, int, int, int]]:
    # Reference: Dalal & Triggs HOG (CVPR 2005) https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf
    out: list[tuple[int, int, int, int]] = []
    roi = frame_bgr
    rh, rw = roi.shape[:2]
    if rw < 64 or rh < 128:
        return out

    scale = 1.0
    target_w = 640
    if rw > target_w:
        scale = float(target_w) / float(rw)
        roi = cv2.resize(roi, (int(round(rw * scale)), int(round(rh * scale))))

    rects, _ = hog.detectMultiScale(
        roi,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05,
    )

    rect_list = []
    if rects is not None and len(rects) > 0:
        rect_list = [list(map(int, r)) for r in rects]

    if rect_list:
        grouped, _ = cv2.groupRectangles(rect_list, groupThreshold=1, eps=0.2)
        rect_list = [list(map(int, r)) for r in grouped] if len(grouped) > 0 else rect_list

    inv = 1.0 / scale
    for (rx, ry, rw2, rh2) in rect_list:
        fx = int(round(rx * inv))
        fy = int(round(ry * inv))
        fw = int(round(rw2 * inv))
        fh = int(round(rh2 * inv))

        px = int(round(fw * 0.05))
        py = int(round(fh * 0.08))

        fx1 = max(fx - px, 0)
        fy1 = max(fy - py, 0)
        fx2 = min(fx + fw + px, int(width))
        fy2 = min(fy + fh + py, int(height))

        if fx2 > fx1 and fy2 > fy1:
            out.append((int(fx1), int(fy1), int(fx2 - fx1), int(fy2 - fy1)))

    return out


def _frame_to_records(
    frame_idx: int,
    bboxes: list[tuple[int, int, int, int]],
    algo: str,
    presence_probability: float | None,
) -> FrameRecord:
    # We record per-frame presence probability to compare against GT presence segments.
    # This is optional and only present when probabilistic tracking is enabled.
    detections = [
        {
            "class": "person",
            "imageAlgorithm": algo,
            "xywh": [int(x), int(y), int(w), int(h)],
        }
        for (x, y, w, h) in bboxes
    ]
    return FrameRecord(
        frame_number=frame_idx,
        detections=detections,
        presence_probability=presence_probability,
    )


def _export_video(
    *,
    video_path: Path,
    output_path: Path,
    motion_cfg: MotionConfig,
    detector_mode: str,
    dnn_model_path: Path | None,
    max_frames: int | None,
    prob: bool,
) -> None:
    cap = open_capture(video_path)
    info = get_video_info(cap)

    detector = MotionDetector(motion_cfg)
    prob_tracker = ProbabilisticPresenceTracker() if prob else None

    hog = None
    if detector_mode == "hog":
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    person_interval = 3 if detector_mode == "hog" else 5
    last_person_frame_idx = -10_000
    last_person_bboxes: list[tuple[int, int, int, int]] = []

    stable_motion = False
    on_counter = 0
    off_counter = 0

    frames: list[dict[str, Any]] = []
    frame_idx = -1
    processed = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_idx += 1
            res = detector.detect(frame)

            on_frames = max(int(motion_cfg.motion_on_frames), 1)
            off_frames = max(int(motion_cfg.motion_off_frames), 1)

            raw_motion = bool(res.motion)
            if not stable_motion:
                if raw_motion:
                    on_counter += 1
                    if on_counter >= on_frames:
                        stable_motion = True
                        off_counter = 0
                else:
                    on_counter = 0
            else:
                if raw_motion:
                    off_counter = 0
                else:
                    off_counter += 1
                    if off_counter >= off_frames:
                        stable_motion = False
                        on_counter = 0

            if stable_motion and raw_motion and detector_mode in ("hog", "dnn"):
                if (frame_idx - last_person_frame_idx) >= person_interval:
                    if detector_mode == "hog" and hog is not None:
                        last_person_bboxes = _hog_people(frame, info.width, info.height, hog)
                    elif detector_mode == "dnn" and dnn_model_path is not None:
                        last_person_bboxes = detect_people_dnn(frame, dnn_model_path)
                    else:
                        last_person_bboxes = []
                    last_person_frame_idx = frame_idx
            else:
                last_person_bboxes = []

            if prob_tracker is not None:
                detected_bbox = None
                if detector_mode in ("hog", "dnn") and last_person_bboxes and raw_motion:
                    candidate = last_person_bboxes
                elif stable_motion and raw_motion:
                    candidate = res.bboxes
                else:
                    candidate = []

                if candidate:
                    detected_bbox = candidate[0] if len(candidate) == 1 else _union_xywh(candidate)

                _ = prob_tracker.update(detected_bbox=detected_bbox)
                draw_motion = prob_tracker.should_draw()
                draw_bbox = prob_tracker.bbox()
                bboxes = [draw_bbox] if (draw_motion and draw_bbox is not None) else []
                presence_probability = prob_tracker.probability() if prob_tracker is not None else None
                record = _frame_to_records(frame_idx, bboxes, detector_mode, presence_probability)
            else:
                if stable_motion and raw_motion:
                    if detector_mode in ("hog", "dnn") and last_person_bboxes:
                        bboxes = last_person_bboxes
                    else:
                        bboxes = res.bboxes
                else:
                    bboxes = []
                record = _frame_to_records(frame_idx, bboxes, detector_mode, None)

            frame_payload: dict[str, Any] = {
                "frameNumber": record.frame_number,
                "detections": record.detections,
            }
            if record.presence_probability is not None:
                frame_payload["presence_probability"] = float(record.presence_probability)
            frames.append(frame_payload)

            processed += 1
            if max_frames is not None and processed >= max_frames:
                break
    finally:
        cap.release()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"frames": frames}, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export per-frame predictions for WiseNET videos.")
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("docs") / "project" / "wisenet_split.json",
        help="Split JSON path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluations") / "predictions",
        help="Base output directory for prediction JSON files.",
    )
    parser.add_argument(
        "--detector",
        choices=["hog", "dnn", "motion"],
        default="hog",
        help="Detector to export (hog/dnn/motion).",
    )
    parser.add_argument("--max-videos", type=int, default=None, help="Limit number of videos.")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames per video.")
    parser.add_argument("--prob", action="store_true", help="Enable probabilistic smoothing.")

    args = parser.parse_args()
    test_items = _load_test_items(args.split)
    videos = list(_iter_videos(test_items))
    if args.max_videos is not None:
        videos = videos[: max(0, int(args.max_videos))]

    dnn_model_path = None
    if args.detector == "dnn":
        dnn_model_path = find_default_dnn_model_path(Path(__file__).resolve().parents[1])
        if dnn_model_path is None:
            raise SystemExit("Default MobileNet-SSD model not found in models/.")

    motion_cfg = MotionConfig(
        min_contour_area=HOG_TUNING_DEFAULTS.min_contour_area,
        blur_ksize=HOG_TUNING_DEFAULTS.blur_ksize,
        mog2_history=HOG_TUNING_DEFAULTS.mog2_history,
        mog2_var_threshold=HOG_TUNING_DEFAULTS.mog2_var_threshold,
        mog2_detect_shadows=HOG_TUNING_DEFAULTS.mog2_detect_shadows,
        mog2_learning_rate=HOG_TUNING_DEFAULTS.mog2_learning_rate,
        threshold_value=HOG_TUNING_DEFAULTS.threshold_value,
        morph_kernel=HOG_TUNING_DEFAULTS.morph_kernel,
        morph_close_iterations=HOG_TUNING_DEFAULTS.morph_close_iterations,
        dilate_iterations=HOG_TUNING_DEFAULTS.dilate_iterations,
        merge_bbox_padding=HOG_TUNING_DEFAULTS.merge_bbox_padding,
        use_connected_components=HOG_TUNING_DEFAULTS.use_connected_components,
        single_box=HOG_TUNING_DEFAULTS.single_box,
        motion_on_frames=HOG_TUNING_DEFAULTS.motion_on_frames,
        motion_off_frames=HOG_TUNING_DEFAULTS.motion_off_frames,
        prob_presence_draw=bool(args.prob),
        roi_rects=(),
    )

    for video_path in videos:
        if not video_path.exists():
            continue
        out_path = args.output / args.detector / f"{video_path.stem}.json"
        _export_video(
            video_path=video_path,
            output_path=out_path,
            motion_cfg=motion_cfg,
            detector_mode=args.detector,
            dnn_model_path=dnn_model_path,
            max_frames=args.max_frames,
            prob=bool(args.prob),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
