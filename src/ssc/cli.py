from __future__ import annotations

import argparse
from pathlib import Path

from ssc.config import MotionConfig, RecorderConfig
from ssc.pipeline import process_video


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CSY3058 AS2 - Smart Security Camera (MVP)")

    p.add_argument("--input", required=True, help="Path to input video (mp4/avi)")
    p.add_argument("--output", default="outputs", help="Output folder for incident clips")

    p.add_argument("--min-area", type=int, default=800, help="Min contour area to count as motion")
    # tip: if it misses motion, lower min-area; if it triggers too often, increase it
    p.add_argument("--post-frames", type=int, default=30, help="Keep recording this many frames after motion stops")

    p.add_argument("--close-iters", type=int, default=2, help="Morphological close iterations (fills gaps in mask)")
    p.add_argument("--dilate-iters", type=int, default=2, help="Dilation iterations (connects blobs; lower = more separate people)")
    p.add_argument("--merge-pad", type=int, default=12, help="Padding when merging nearby boxes")
    p.add_argument(
        "--learning-rate",
        type=float,
        default=-1.0,
        help="MOG2 learningRate (-1 auto, 0=no update, small values update slowly)",
    )

    p.add_argument(
        "--on-frames",
        type=int,
        default=2,
        help="Require this many consecutive motion frames before triggering (reduces 1-frame ghosts)",
    )
    p.add_argument(
        "--off-frames",
        type=int,
        default=3,
        help="Require this many consecutive no-motion frames before clearing motion (reduces flicker)",
    )

    p.add_argument(
        "--single-box",
        action="store_true",
        help="Force a single merged bbox per frame (useful when there is one person)",
    )

    p.add_argument(
        "--cc",
        action="store_true",
        help="Use connected components for blob extraction (optional alternative)",
    )

    p.add_argument(
        "--prob-draw",
        action="store_true",
        help="Use probabilistic presence smoothing for drawing only (recording unchanged)",
    )

    p.add_argument(
        "--roi",
        action="append",
        default=[],
        help="ROI rectangle in pixels as x1,y1,x2,y2 (repeatable). Areas outside ROIs are ignored.",
    )

    p.add_argument("--display", action="store_true", help="Show video preview window")
    p.add_argument("--show-mask", action="store_true", help="Show the motion mask window")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    input_path = Path(args.input)
    output_dir = Path(args.output)

    roi_rects: list[tuple[int, int, int, int]] = []
    for raw in getattr(args, "roi", []) or []:
        parts = [p.strip() for p in str(raw).split(",")]
        if len(parts) != 4:
            raise SystemExit(f"--roi must be x1,y1,x2,y2 (got: {raw})")
        try:
            x1, y1, x2, y2 = (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]))
        except ValueError:
            raise SystemExit(f"--roi must contain integers (got: {raw})")
        roi_rects.append((x1, y1, x2, y2))

    motion_cfg = MotionConfig(
        min_contour_area=args.min_area,
        morph_close_iterations=args.close_iters,
        dilate_iterations=args.dilate_iters,
        merge_bbox_padding=args.merge_pad,
        mog2_learning_rate=args.learning_rate,
        use_connected_components=bool(args.cc),
        single_box=bool(args.single_box),
        motion_on_frames=args.on_frames,
        motion_off_frames=args.off_frames,
        prob_presence_draw=bool(args.prob_draw),
        roi_rects=tuple(roi_rects),
    )
    recorder_cfg = RecorderConfig(post_event_frames=args.post_frames)

    return process_video(
        input_path=input_path,
        output_dir=output_dir,
        motion_cfg=motion_cfg,
        recorder_cfg=recorder_cfg,
        display=bool(args.display),
        show_mask=bool(args.show_mask),
    )
