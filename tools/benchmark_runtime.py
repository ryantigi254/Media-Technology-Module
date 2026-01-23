"""Runtime benchmark for the smart security camera pipeline.

This script was generated with AI assistance and is intended to provide a repeatable,
transparent way to measure throughput (FPS) for the different detection backends.

It benchmarks the *analytics* path (decode + motion gate + optional person model),
and intentionally avoids recorder/overlay I/O so the reported FPS reflects the
core processing cost.
"""

from __future__ import annotations


import argparse
import json
import platform
import time
from dataclasses import asdict
import os
from pathlib import Path

import cv2


PERSON_INTERVAL_HOG = 3
PERSON_INTERVAL_DNN = 5
PERSON_INTERVAL_YOLO = 5


def _bootstrap_src_path() -> None:
    # allows running as: python tools/benchmark_runtime.py
    here = Path(__file__).resolve()
    root = here.parent.parent
    src = root / "src"
    import sys

    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_bootstrap_src_path()


from ssc.config import HOG_TUNING_DEFAULTS, MotionConfig
from ssc.features.dnn_person_detector import detect_people_dnn
from ssc.features.yolo_person_detector import detect_objects_yolo, is_yolo_available
from ssc.motion import MotionDetector


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark throughput (FPS) for SSC backends")
    p.add_argument("--video", type=Path, required=True, help="Path to a video file")
    p.add_argument(
        "--max-frames",
        type=int,
        default=600,
        help="Max frames to process (after warmup is included in this cap)",
    )
    p.add_argument(
        "--warmup-frames",
        type=int,
        default=30,
        help="Warmup frames excluded from timing (helps exclude model load)",
    )
    p.add_argument(
        "--motion-config",
        type=str,
        default="hog_defaults",
        choices=["default", "hog_defaults"],
        help="Motion config preset to use",
    )
    p.add_argument(
        "--modes",
        type=str,
        default="motion,hog,dnn,yolo",
        help="Comma-separated list: motion,hog,dnn,yolo",
    )
    p.add_argument(
        "--dnn-model",
        type=Path,
        default=Path("models/mobilenet_ssd/mobilenet_ssd.caffemodel"),
        help="Path to MobileNet-SSD .caffemodel (expects matching .prototxt alongside)",
    )
    p.add_argument(
        "--yolo-model",
        type=str,
        default="yolov8n.pt",
        help="Ultralytics model name/path (e.g. yolov8n.pt)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("evaluations/meta/runtime_benchmark.json"),
        help="Where to write JSON results",
    )
    return p.parse_args()


def _select_motion_cfg(preset: str) -> MotionConfig:
    if preset == "default":
        return MotionConfig()
    return HOG_TUNING_DEFAULTS


def _bench_one(
    *,
    video_path: Path,
    mode: str,
    motion_cfg: MotionConfig,
    max_frames: int,
    warmup_frames: int,
    dnn_model_path: Path,
    yolo_model_name: str,
) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    detector = MotionDetector(motion_cfg)

    hog = None
    if mode == "hog":
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    if mode == "hog":
        person_interval = PERSON_INTERVAL_HOG
    elif mode == "dnn":
        person_interval = PERSON_INTERVAL_DNN
    elif mode == "yolo":
        person_interval = PERSON_INTERVAL_YOLO
    else:
        person_interval = PERSON_INTERVAL_DNN

    on_frames = max(int(motion_cfg.motion_on_frames), 1)
    off_frames = max(int(motion_cfg.motion_off_frames), 1)

    stable_motion = False
    on_counter = 0
    off_counter = 0
    frame_idx = -1

    processed_total = 0
    processed_timed = 0
    model_calls = 0
    model_call_time_sec = 0.0

    wall_time_sec = 0.0

    last_person_frame_idx = -10_000

    try:
        while processed_total < max_frames:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break

            frame_idx += 1
            processed_total += 1

            res = detector.detect(frame)
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

            if mode in ("hog", "dnn", "yolo") and stable_motion and raw_motion:
                if (frame_idx - last_person_frame_idx) >= person_interval:
                    if mode == "hog":
                        if hog is not None:
                            m0 = time.perf_counter()
                            _rects, _weights = hog.detectMultiScale(
                                frame,
                                winStride=(8, 8),
                                padding=(8, 8),
                                scale=1.05,
                            )
                            m1 = time.perf_counter()
                            model_calls += 1
                            model_call_time_sec += float(m1 - m0)
                    elif mode == "dnn":
                        m0 = time.perf_counter()
                        _ = detect_people_dnn(frame, dnn_model_path)
                        m1 = time.perf_counter()
                        model_calls += 1
                        model_call_time_sec += float(m1 - m0)
                    else:
                        m0 = time.perf_counter()
                        _ = detect_objects_yolo(frame, model_name=yolo_model_name)
                        m1 = time.perf_counter()
                        model_calls += 1
                        model_call_time_sec += float(m1 - m0)

                    last_person_frame_idx = frame_idx

            t1 = time.perf_counter()

            # exclude warmup frames from timing to reduce first-call model load noise
            if processed_total > warmup_frames:
                processed_timed += 1
                wall_time_sec += float(t1 - t0)

    finally:
        cap.release()

    fps = (float(processed_timed) / wall_time_sec) if wall_time_sec > 0 and processed_timed > 0 else 0.0
    avg_model_call_ms = (
        (float(model_call_time_sec) / float(model_calls) * 1000.0) if model_calls > 0 else 0.0
    )

    return {
        "mode": mode,
        "processed_total": int(processed_total),
        "processed_timed": int(processed_timed),
        "warmup_frames": int(warmup_frames),
        "wall_time_sec": float(wall_time_sec),
        "fps": float(fps),
        "model_calls": int(model_calls),
        "model_call_time_sec": float(model_call_time_sec),
        "avg_model_call_ms": float(avg_model_call_ms),
        "model_calls_per_100_frames": float(model_calls) / float(processed_total) * 100.0 if processed_total else 0.0,
    }


def main() -> int:
    args = _parse_args()

    project_root = Path(__file__).resolve().parent.parent

    video_path = Path(args.video)
    if not video_path.is_file():
        raise SystemExit(f"Video does not exist: {video_path}")

    modes = [m.strip().lower() for m in str(args.modes).split(",") if m.strip()]
    allowed = {"motion", "hog", "dnn", "yolo"}
    modes = [m for m in modes if m in allowed]
    if not modes:
        raise SystemExit("No valid modes requested")

    motion_cfg = _select_motion_cfg(str(args.motion_config))

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = project_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dnn_model_path = Path(args.dnn_model)
    if not dnn_model_path.is_absolute():
        dnn_model_path = project_root / dnn_model_path

    yolo_model_name = str(args.yolo_model)
    # If a relative YOLO path is provided, resolve it to the project root.
    yolo_model_path = Path(yolo_model_name)
    if not yolo_model_path.is_absolute() and (project_root / yolo_model_path).exists():
        yolo_model_name = str((project_root / yolo_model_path).resolve())

    env = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "opencv": getattr(cv2, "__version__", "unknown"),
        "cpu_count": int(os.cpu_count() or 0),
        "yolo_available": bool(is_yolo_available()),
    }

    results = []
    for mode in modes:
        if mode == "yolo" and not is_yolo_available():
            results.append({
                "mode": "yolo",
                "skipped": True,
                "reason": "ultralytics not installed in this environment",
            })
            continue

        results.append(
            _bench_one(
                video_path=video_path,
                mode=mode,
                motion_cfg=motion_cfg,
                max_frames=int(args.max_frames),
                warmup_frames=int(args.warmup_frames),
                dnn_model_path=dnn_model_path,
                yolo_model_name=yolo_model_name,
            )
        )

    payload = {
        "video": str(video_path),
        "motion_config": asdict(motion_cfg),
        "env": env,
        "results": results,
    }

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # human-friendly output
    print(f"Video: {video_path}")
    print(f"Warmup frames: {args.warmup_frames} | Max frames: {args.max_frames}")
    print(f"OpenCV: {env['opencv']} | Python: {env['python']} | YOLO available: {env['yolo_available']}")
    for r in results:
        if r.get("skipped"):
            print(f"{r['mode']}: SKIPPED ({r['reason']})")
            continue
        print(
            f"{r['mode']}: {r['fps']:.2f} FPS | model calls: {r['model_calls']} "
            f"({r['model_calls_per_100_frames']:.1f}/100 frames)"
        )
    print(f"Wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
