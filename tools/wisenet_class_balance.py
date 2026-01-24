from __future__ import annotations

# This file was AI assisted as it is mostly for evaluations and testing purposes.

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import cv2


def _load_split(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "test" in payload:
        return payload
    if "split" in payload and "test" in payload["split"]:
        return payload["split"]
    raise ValueError("Split JSON does not contain a test split.")


def _extract_frame_numbers(annotation_path: Path) -> List[int]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    frames = payload.get("frames", []) if isinstance(payload, dict) else []
    frame_numbers: List[int] = []
    for frame_obj in frames:
        if not isinstance(frame_obj, dict):
            continue
        if "frameNumber" in frame_obj:
            frame_numbers.append(int(frame_obj["frameNumber"]))
        elif "frame_index" in frame_obj:
            frame_numbers.append(int(frame_obj["frame_index"]))
        elif "frame" in frame_obj:
            frame_numbers.append(int(frame_obj["frame"]))
    return frame_numbers


def _open_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return max(0, total)
    finally:
        cap.release()


def _resolve_annotation_path(annotation_root: Path, video_path: Path) -> Path:
    for part in video_path.parts:
        if part.lower().startswith("set_"):
            candidate = annotation_root / part / f"{video_path.stem}.json"
            if candidate.exists():
                return candidate
    fallback = annotation_root / f"{video_path.stem}.json"
    return fallback


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute class balance on WiseNET test split.")
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("docs") / "project" / "wisenet_split.json",
        help="Split JSON path.",
    )
    parser.add_argument(
        "--annotation-root",
        type=Path,
        default=Path("data") / "wisenet" / "manual_annotations" / "people_detection",
        help="Manual annotation root.",
    )
        parser.add_argument(
            "--out",
            type=Path,
            default=Path("evaluations") / "meta" / "class_balance.json",
            help="Output JSON path.",
        )

    args = parser.parse_args()
    split_payload = _load_split(args.split)
    test_items = split_payload["test"]

    results: List[Dict[str, Any]] = []
    for item in test_items:
        video_path = Path(item["video"] if "video" in item else item["video_path"])
        annotation_path = _resolve_annotation_path(args.annotation_root, video_path)
        if not video_path.exists() or not annotation_path.exists():
            continue

        total_frames = _open_frame_count(video_path)
        positive_frames = len(_extract_frame_numbers(annotation_path))
        negative_frames = max(0, total_frames - positive_frames)
        negative_ratio = float(negative_frames / total_frames) if total_frames > 0 else 0.0

        results.append(
            {
                "video": str(video_path),
                "total_frames": total_frames,
                "positive_frames": positive_frames,
                "negative_frames": negative_frames,
                "negative_ratio": negative_ratio,
            }
        )

    summary = {
        "videos": results,
        "aggregate": {
            "total_frames": sum(r["total_frames"] for r in results),
            "positive_frames": sum(r["positive_frames"] for r in results),
            "negative_frames": sum(r["negative_frames"] for r in results),
        },
    }
    if summary["aggregate"]["total_frames"] > 0:
        summary["aggregate"]["negative_ratio"] = float(
            summary["aggregate"]["negative_frames"] / summary["aggregate"]["total_frames"]
        )
    else:
        summary["aggregate"]["negative_ratio"] = 0.0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
