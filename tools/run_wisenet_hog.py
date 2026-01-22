from __future__ import annotations

# This file was AI assisted as it is mostly for evaluations and testing purposes.

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from ssc.config import HOG_TUNING_DEFAULTS, MotionConfig, RecorderConfig
from ssc.pipeline import process_video


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run WiseNET test split with HOG settings.")
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("docs") / "project" / "wisenet_split.json",
        help="Split JSON path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / "wisenet_hog",
        help="Base output directory for HOG runs.",
    )
    parser.add_argument("--max-videos", type=int, default=None, help="Limit number of videos.")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames per video.")
    parser.add_argument("--prob", action="store_true", help="Enable probabilistic smoothing.")

    args = parser.parse_args()
    split_path = args.split
    output_dir = args.output

    test_items = _load_test_items(split_path)
    videos = list(_iter_videos(test_items))
    if args.max_videos is not None:
        videos = videos[: max(0, int(args.max_videos))]

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
    recorder_cfg = RecorderConfig(post_event_frames=30)

    for video_path in videos:
        if not video_path.exists():
            continue
        process_video(
            input_path=video_path,
            output_dir=output_dir,
            motion_cfg=motion_cfg,
            recorder_cfg=recorder_cfg,
            display=False,
            show_mask=False,
            max_frames=args.max_frames,
            person_detector="hog",
            dnn_model_path=None,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
