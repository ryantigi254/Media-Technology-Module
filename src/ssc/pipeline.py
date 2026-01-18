from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

import cv2

from ssc.config import MotionConfig, RecorderConfig
from ssc.features.probabilistic_presence import ProbabilisticPresenceTracker
from ssc.features.dnn_person_detector import detect_people_dnn
from ssc.features.yolo_person_detector import detect_objects_yolo
from ssc.motion import MotionDetector
from ssc.recorder import IncidentRecorder
from ssc.video_io import get_video_info, open_capture


def _annotate(
    frame_bgr,
    frame_idx: int,
    bboxes: list[tuple[int, int, int, int]],
    motion: bool,
    presence_p: float | None = None,
):
    out = frame_bgr.copy()

    frame_h, frame_w = out.shape[:2]

    for (x, y, bw, bh) in bboxes:
        # bbox merging adds padding so coordinates can go outside the frame a bit
        x1 = max(int(x), 0)
        y1 = max(int(y), 0)
        x2 = min(int(x + bw), frame_w - 1)
        y2 = min(int(y + bh), frame_h - 1)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

    label = "MOTION" if motion else "idle"
    cv2.putText(
        out,
        f"{label} | frame={frame_idx}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0) if motion else (200, 200, 200),
        2,
        cv2.LINE_AA,
    )

    if presence_p is not None:
        cv2.putText(
            out,
            f"P(presence)={presence_p:.2f}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if motion else (200, 200, 200),
            2,
            cv2.LINE_AA,
        )

    return out


PERSON_INTERVAL_HOG = 3
PERSON_INTERVAL_DNN = 5
PERSON_INTERVAL_YOLO = 5

# The motion/YOLO gating alignment was AI-assisted to mirror GUI and CLI behaviour.


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


def _filter_to_roi(
    bboxes: list[tuple[int, int, int, int]],
    roi_rects: tuple[tuple[int, int, int, int], ...],
) -> list[tuple[int, int, int, int]]:
    if not roi_rects:
        return list(bboxes)
    filtered: list[tuple[int, int, int, int]] = []
    for x, y, w, h in bboxes:
        cx = float(x) + float(w) * 0.5
        cy = float(y) + float(h) * 0.5
        for rx1, ry1, rx2, ry2 in roi_rects:
            if float(rx1) <= cx <= float(rx2) and float(ry1) <= cy <= float(ry2):
                # Clip box to this ROI
                nx1 = max(int(x), int(rx1))
                ny1 = max(int(y), int(ry1))
                nx2 = min(int(x + w), int(rx2))
                ny2 = min(int(y + h), int(ry2))
                
                if nx2 > nx1 and ny2 > ny1:
                    filtered.append((nx1, ny1, nx2 - nx1, ny2 - ny1))
                break
    return filtered


def _safe_name(s: str) -> str:
    keep = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep)
    return out.strip("_") or "unknown"


def _guess_set_name(input_path: Path) -> str:
    parts = [p.lower() for p in input_path.parts]
    try:
        i = parts.index("data")
        if i + 1 < len(input_path.parts):
            return _safe_name(input_path.parts[i + 1])
    except ValueError:
        pass

    parent = input_path.parent.name
    return _safe_name(parent)


def _make_run_dir(*, base_output_dir: Path, input_path: Path) -> Path:
    set_name = _guess_set_name(input_path)
    video_name = _safe_name(input_path.stem)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_output_dir / set_name / video_name / f"run_{stamp}"


def process_video(
    *,
    input_path: Path,
    output_dir: Path,
    motion_cfg: MotionConfig,
    recorder_cfg: RecorderConfig,
    display: bool,
    show_mask: bool,
    stop_event: threading.Event | None = None,
    max_frames: int | None = None,
    person_detector: str = "motion",
    dnn_model_path: Path | None = None,
) -> int:
    cap = open_capture(input_path)
    info = get_video_info(cap)

    run_dir = _make_run_dir(base_output_dir=output_dir, input_path=input_path)
    run_dir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "input_path": str(input_path),
        "output_dir": str(run_dir),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "video_info": {
            "fps": info.fps,
            "width": info.width,
            "height": info.height,
            "frame_count": info.frame_count,
        },
        "motion_cfg": motion_cfg.__dict__,
        "recorder_cfg": recorder_cfg.__dict__,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    # detector finds motion blobs; recorder decides when to start/stop incident clips
    detector = MotionDetector(motion_cfg)
    recorder = IncidentRecorder(output_dir=run_dir, video_info=info, cfg=recorder_cfg)

    prob_tracker = ProbabilisticPresenceTracker() if (motion_cfg.enable_smoothing or motion_cfg.show_confidence) else None

    detector_mode = (person_detector or "motion").strip().lower()
    if detector_mode not in ("hog", "dnn", "yolo", "motion"):
        detector_mode = "motion"

    if detector_mode == "dnn":
        if (
            dnn_model_path is None
            or not dnn_model_path.is_file()
            or not dnn_model_path.with_suffix(".prototxt").is_file()
            or not dnn_model_path.with_suffix(".caffemodel").is_file()
        ):
            raise ValueError("DNN mode requires a valid .prototxt/.caffemodel model path.")

    hog = None
    if detector_mode == "hog":
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    if detector_mode == "hog":
        person_interval = PERSON_INTERVAL_HOG
    elif detector_mode == "dnn":
        person_interval = PERSON_INTERVAL_DNN
    elif detector_mode == "yolo":
        person_interval = PERSON_INTERVAL_YOLO
    else:
        person_interval = PERSON_INTERVAL_DNN
    last_person_frame_idx = -10_000
    last_person_bboxes: list[tuple[int, int, int, int]] = []

    frame_idx = -1
    processed = 0
    motion_frames = 0
    stable_motion = False
    on_counter = 0
    off_counter = 0
    cancelled = False
    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                cancelled = True
                break

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

            effective_motion = stable_motion and raw_motion

            if detector_mode == "yolo":
                if (frame_idx - last_person_frame_idx) >= person_interval:
                    last_person_bboxes = detect_objects_yolo(frame)
                    last_person_bboxes = _filter_to_roi(last_person_bboxes, motion_cfg.roi_rects)
                    last_person_frame_idx = frame_idx
            elif stable_motion and raw_motion and detector_mode in ("hog", "dnn"):
                if (frame_idx - last_person_frame_idx) >= person_interval:
                    if detector_mode == "hog":
                        last_person_bboxes = []
                        if hog is not None:
                            rois = list(motion_cfg.roi_rects)
                            if not rois:
                                rois = [(0, 0, int(info.width), int(info.height))]
                            for x1, y1, x2, y2 in rois:
                                ix1 = max(int(x1), 0)
                                iy1 = max(int(y1), 0)
                                ix2 = min(int(x2), int(info.width))
                                iy2 = min(int(y2), int(info.height))
                                if ix2 <= ix1 or iy2 <= iy1:
                                    continue

                                roi = frame[iy1:iy2, ix1:ix2]
                                if roi.size == 0:
                                    continue

                                rh, rw = roi.shape[:2]
                                if rw < 64 or rh < 128:
                                    continue

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

                                    fx1 = max(ix1 + fx - px, 0)
                                    fy1 = max(iy1 + fy - py, 0)
                                    fx2 = min(ix1 + fx + fw + px, int(info.width))
                                    fy2 = min(iy1 + fy + fh + py, int(info.height))

                                    if fx2 > fx1 and fy2 > fy1:
                                        last_person_bboxes.append(
                                            (int(fx1), int(fy1), int(fx2 - fx1), int(fy2 - fy1))
                                        )
                    else:
                        assert dnn_model_path is not None
                        last_person_bboxes = detect_people_dnn(frame, dnn_model_path)
                        last_person_bboxes = _filter_to_roi(last_person_bboxes, motion_cfg.roi_rects)
                    last_person_frame_idx = frame_idx
            elif detector_mode != "yolo":
                last_person_bboxes = []

            # --- Selection Logic ---
            p_val: float | None = None
            display_bboxes: list[tuple[int, int, int, int]] = []
            
            # 1. Update Tracker (if active for either smoothing or confidence)
            if prob_tracker is not None:
                detected_bbox = None
                # Tracker input selection
                if detector_mode == "yolo" and last_person_bboxes:
                    candidate = last_person_bboxes
                elif detector_mode in ("hog", "dnn") and last_person_bboxes and raw_motion:
                    candidate = _filter_to_roi(last_person_bboxes, motion_cfg.roi_rects)
                elif effective_motion:
                    candidate = _filter_to_roi(res.bboxes, motion_cfg.roi_rects)
                else:
                    candidate = []

                if candidate:
                    detected_bbox = candidate[0] if len(candidate) == 1 else _union_xywh(candidate)

                _ = prob_tracker.update(detected_bbox=detected_bbox)
                if motion_cfg.show_confidence:
                    p_val = prob_tracker.probability()

            # 2. Decide Display Boxes
            if motion_cfg.enable_smoothing and prob_tracker is not None:
                # Use Smoothed Boxes
                draw_motion = prob_tracker.should_draw()
                draw_bbox = prob_tracker.bbox()
                display_bboxes = [draw_bbox] if (draw_motion and draw_bbox is not None) else []
            else:
                # Use Raw Boxes
                if detector_mode == "yolo" and last_person_bboxes:
                    display_bboxes = last_person_bboxes
                elif stable_motion and raw_motion:
                    if detector_mode in ("hog", "dnn") and last_person_bboxes:
                        display_bboxes = _filter_to_roi(last_person_bboxes, motion_cfg.roi_rects)
                    else:
                        display_bboxes = _filter_to_roi(res.bboxes, motion_cfg.roi_rects)
                else:
                    display_bboxes = []
            
            # 3. Annotate
            should_draw_motion = bool(display_bboxes) or (stable_motion and raw_motion)
            annotated = _annotate(frame, frame_idx, display_bboxes, should_draw_motion, presence_p=p_val)

            recorder.update(frame_idx, annotated, motion=effective_motion)

            if effective_motion:
                motion_frames += 1

            if display:
                cv2.imshow("smart-security-camera", annotated)
                if show_mask:
                    cv2.imshow("motion-mask", res.motion_mask)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            processed += 1
            # max_frames is mainly for unit/integration tests so they don't run for ages
            if max_frames is not None and processed >= max_frames:
                break

    finally:
        recorder.close(final_frame_idx=frame_idx)
        cap.release()
        if display:
            cv2.destroyAllWindows()

        durations_frames = recorder.incident_durations_frames
        incident_count = len(durations_frames)
        durations_sec = [float(frames) / float(info.fps) for frames in durations_frames if info.fps > 0]
        avg_duration_sec = float(sum(durations_sec) / len(durations_sec)) if durations_sec else 0.0
        max_duration_sec = float(max(durations_sec)) if durations_sec else 0.0

        startup_window_frames = int(round(float(info.fps))) if info.fps > 0 else 0
        first_start = recorder.first_incident_start_frame
        false_start_incident = False
        if (
            first_start is not None
            and startup_window_frames > 0
            and first_start < startup_window_frames
            and durations_frames
            and int(durations_frames[0]) < startup_window_frames
        ):
            false_start_incident = True

        run_summary = {
            "processed_frames": int(processed),
            "motion_frames": int(motion_frames),
            "motion_frames_ratio": (float(motion_frames) / float(processed)) if processed > 0 else 0.0,
            "incident_count": int(incident_count),
            "avg_incident_duration_sec": float(avg_duration_sec),
            "max_incident_duration_sec": float(max_duration_sec),
            "startup_window_frames": int(startup_window_frames),
            "false_start_incident": bool(false_start_incident),
            "cancelled": bool(cancelled),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        }

        (run_dir / "run_summary.json").write_text(
            json.dumps(run_summary, indent=2), encoding="utf-8"
        )

        run_meta["run_summary"] = run_summary
        (run_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    return 1 if cancelled else 0
