from __future__ import annotations

# This file was AI assisted as it is mostly for evaluations and testing purposes.

import argparse
import json
import random
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


@dataclass(frozen=True)
class VideoItem:
    video_path: Path
    annotation_path: Path | None
    set_name: str


def _find_videos(video_sets_dir: Path) -> list[Path]:
    videos: list[Path] = []
    for path in video_sets_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
            videos.append(path)
    return sorted(videos)


def _index_annotations(manual_dir: Path) -> dict[str, list[Path]]:
    by_stem: dict[str, list[Path]] = {}
    if not manual_dir.exists():
        return by_stem

    for path in manual_dir.rglob("*.json"):
        by_stem.setdefault(path.stem, []).append(path)
    return by_stem


def _match_annotation(video: Path, ann_by_stem: dict[str, list[Path]]) -> Path | None:
    if video.stem in ann_by_stem:
        return sorted(ann_by_stem[video.stem], key=lambda p: str(p).lower())[0]
    return None


def _extract_set_name(video_sets_dir: Path, video_path: Path) -> str:
    try:
        rel = video_path.relative_to(video_sets_dir)
    except ValueError:
        return video_path.parent.name
    parts = rel.parts
    return parts[0] if parts else video_path.parent.name


def _build_items(video_sets_dir: Path, manual_dir: Path) -> list[VideoItem]:
    ann_by_stem = _index_annotations(manual_dir)
    items: list[VideoItem] = []
    for video in _find_videos(video_sets_dir):
        ann = _match_annotation(video, ann_by_stem)
        set_name = _extract_set_name(video_sets_dir, video)
        items.append(VideoItem(video_path=video, annotation_path=ann, set_name=set_name))
    return items


def _split_by_set(items: list[VideoItem], test_ratio: float, seed: int) -> tuple[list[VideoItem], list[VideoItem]]:
    rng = random.Random(seed)
    sets = sorted({item.set_name for item in items})
    rng.shuffle(sets)
    test_count = max(1, int(round(len(sets) * test_ratio))) if sets else 0
    test_sets = set(sets[:test_count])
    train = [item for item in items if item.set_name not in test_sets]
    test = [item for item in items if item.set_name in test_sets]
    return train, test


def _split_by_video(items: list[VideoItem], test_ratio: float, seed: int) -> tuple[list[VideoItem], list[VideoItem]]:
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    test_count = max(1, int(round(len(shuffled) * test_ratio))) if shuffled else 0
    test = shuffled[:test_count]
    train = shuffled[test_count:]
    return train, test


def _to_record(item: VideoItem) -> dict[str, str | None]:
    return {
        "video": str(item.video_path),
        "annotation": str(item.annotation_path) if item.annotation_path is not None else None,
        "set": item.set_name,
    }


def _write_split(out_path: Path, train: Iterable[VideoItem], test: Iterable[VideoItem]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train": [_to_record(item) for item in train],
        "test": [_to_record(item) for item in test],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _download_kaggle(dataset: str, dest: Path) -> None:
    if shutil.which("kaggle") is None:
        raise RuntimeError("Kaggle CLI not found in PATH.")
    dest.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", dataset, "-p", str(dest), "--unzip"]
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a train/test split for WiseNET dataset.")
    parser.add_argument(
        "--dataset",
        default="abdelrhmannile/wisenet",
        help="Kaggle dataset slug (owner/dataset)",
    )
    parser.add_argument(
        "--data-dir",
        default=str(Path("data") / "wisenet"),
        help="Local dataset folder (will contain video_sets/ and manual_annotations/)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download dataset via Kaggle CLI into data-dir.",
    )
    parser.add_argument(
        "--split-by",
        choices=["set", "video"],
        default="set",
        help="Split by set (recommended) or by video.",
    )
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=3058, help="Random seed.")
    parser.add_argument(
        "--out",
        default=str(Path("docs") / "project" / "wisenet_split.json"),
        help="Output split JSON path.",
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    if args.download:
        _download_kaggle(args.dataset, data_dir)

    video_sets_dir = data_dir / "video_sets"
    manual_dir = data_dir / "manual_annotations"
    if not video_sets_dir.exists():
        raise SystemExit(f"Missing {video_sets_dir}")

    items = _build_items(video_sets_dir, manual_dir)
    if not items:
        raise SystemExit("No videos found under video_sets/")

    if args.split_by == "set":
        train, test = _split_by_set(items, args.test_ratio, args.seed)
    else:
        train, test = _split_by_video(items, args.test_ratio, args.seed)

    out_path = Path(args.out)
    _write_split(out_path, train, test)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
