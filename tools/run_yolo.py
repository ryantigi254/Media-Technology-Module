from __future__ import annotations

import argparse
import threading
import sys
from pathlib import Path


# This script was heavily AI-generated to provide a more intelligent evaluation runner.
# Reason: fast testing for the demo runs; reviewed and adjusted to match project defaults.


def _bootstrap_src_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main() -> int:
    _bootstrap_src_path()
    from ssc.config import MotionConfig, RecorderConfig
    from ssc.pipeline import process_video

    p = argparse.ArgumentParser(description="Run the pipeline using YOLO person detection (Ultralytics).")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", default=Path("outputs"), type=Path)

    # Motion tuning (keep consistent with existing CLI surface)
    p.add_argument("--min-area", default=800, type=int)
    p.add_argument("--post-frames", default=30, type=int)
    p.add_argument("--close-iters", default=2, type=int)
    p.add_argument("--dilate-iters", default=2, type=int)
    p.add_argument("--merge-pad", default=12, type=int)
    p.add_argument("--learning-rate", default=-1.0, type=float)
    p.add_argument("--on-frames", default=2, type=int)
    p.add_argument("--off-frames", default=3, type=int)
    p.add_argument("--display", action="store_true")
    p.add_argument("--show-mask", action="store_true")
    p.add_argument("--roi", action="append", default=[], help="ROI rect x1,y1,x2,y2 (can repeat)")

    args = p.parse_args()

    roi_rects: list[tuple[int, int, int, int]] = []
    for raw in args.roi or []:
        parts = [p.strip() for p in str(raw).split(",")]
        if len(parts) != 4:
            raise SystemExit(f"--roi must be x1,y1,x2,y2 (got: {raw})")
        try:
            x1, y1, x2, y2 = (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]))
        except ValueError:
            raise SystemExit(f"--roi must contain integers (got: {raw})")
        roi_rects.append((x1, y1, x2, y2))

    motion_cfg = MotionConfig(
        min_contour_area=int(args.min_area),
        morph_close_iterations=int(args.close_iters),
        dilate_iterations=int(args.dilate_iters),
        merge_bbox_padding=int(args.merge_pad),
        mog2_learning_rate=float(args.learning_rate),
        motion_on_frames=int(args.on_frames),
        motion_off_frames=int(args.off_frames),
        roi_rects=tuple(roi_rects),
    )
    recorder_cfg = RecorderConfig(post_event_frames=int(args.post_frames))

    stop_event = threading.Event()
    return process_video(
        input_path=args.input,
        output_dir=args.output,
        motion_cfg=motion_cfg,
        recorder_cfg=recorder_cfg,
        display=bool(args.display),
        show_mask=bool(args.show_mask),
        stop_event=stop_event,
        person_detector="yolo",
    )


if __name__ == "__main__":
    raise SystemExit(main())
