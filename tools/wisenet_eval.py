from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

FrameIndex = int
BBox = Tuple[float, float, float, float]

SET_NAME_PATTERN = re.compile(r"set_\d+", re.IGNORECASE)
PROBABILITY_MATCH_THRESHOLD = 0.5
PERSON_CLASS_FILTER = ("person",)

# Dataset reference:
# - WiseNET (Kaggle): https://www.kaggle.com/datasets/dirkbk/wisenet-cctv-annotations
# The evaluation script was heavily AI-assisted and then verified.
# Reason: it adds sophisticated metrics (continuity, probability alignment, IoU stats) that are not required by the brief,
# and it exists to evaluate our implementation rather than be part of the core system.


@dataclass(frozen=True)
class FrameDetections:
    by_frame: Dict[FrameIndex, List[BBox]]
    probability_by_frame: Dict[FrameIndex, float]


@dataclass(frozen=True)
class VideoMetrics:
    video_name: str
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    presence_accuracy: float
    presence_iou: float
    person_continuity_max_frames: Optional[int]
    person_continuity_rate: Optional[float]
    mean_iou: float
    iou_50_rate: float
    entry_detected: Optional[bool]
    exit_detected: Optional[bool]
    entry_latency_frames: Optional[int]
    exit_latency_frames: Optional[int]
    false_alarms_per_min: Optional[float]
    probability_mean_present: Optional[float]
    probability_mean_absent: Optional[float]
    probability_match_rate: Optional[float]


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den != 0 else 0.0


def _normalise_bbox(raw: Any) -> Optional[BBox]:
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        x, y, w, h = raw
        return float(x), float(y), float(w), float(h)

    if isinstance(raw, dict):
        if all(key in raw for key in ("x", "y", "w", "h")):
            return float(raw["x"]), float(raw["y"]), float(raw["w"]), float(raw["h"])
        if all(key in raw for key in ("xmin", "ymin", "xmax", "ymax")):
            x1 = float(raw["xmin"])
            y1 = float(raw["ymin"])
            x2 = float(raw["xmax"])
            y2 = float(raw["ymax"])
            return x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)
    return None


def _extract_frames(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("frames", "detections", "annotations"):
            if key in payload and isinstance(payload[key], list):
                return payload[key]
    if isinstance(payload, list):
        return payload
    return []


def _extract_frame_index(frame_obj: Dict[str, Any]) -> Optional[int]:
    for key in ("frame_index", "frame", "frame_id", "frame_idx", "frameNumber"):
        if key in frame_obj:
            return int(frame_obj[key])
    return None


def _extract_label(raw: Any) -> Optional[str]:
    if isinstance(raw, dict):
        for key in ("class", "label", "name"):
            if key in raw:
                return str(raw[key]).strip().lower()
    return None


def _extract_bboxes(frame_obj: Dict[str, Any], class_filter: Optional[Tuple[str, ...]]) -> List[BBox]:
    for key in ("bboxes", "boxes", "detections", "objects"):
        if key in frame_obj and isinstance(frame_obj[key], list):
            out: List[BBox] = []
            for raw in frame_obj[key]:
                label = _extract_label(raw)
                if class_filter and label is not None and label not in class_filter:
                    continue
                if isinstance(raw, dict) and "xywh" in raw:
                    bbox = _normalise_bbox(raw.get("xywh"))
                else:
                    bbox = _normalise_bbox(raw)
                if bbox is not None:
                    out.append(bbox)
            return out
    return []


def load_detections(path: Path, class_filter: Optional[Tuple[str, ...]] = PERSON_CLASS_FILTER) -> FrameDetections:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = _extract_frames(payload)
    by_frame: Dict[FrameIndex, List[BBox]] = {}
    probability_by_frame: Dict[FrameIndex, float] = {}
    for frame_obj in frames:
        if not isinstance(frame_obj, dict):
            continue
        frame_index = _extract_frame_index(frame_obj)
        if frame_index is None:
            continue
        bboxes = _extract_bboxes(frame_obj, class_filter)
        by_frame[int(frame_index)] = bboxes
        for key in ("presence_probability", "probability", "presence_p"):
            if key in frame_obj:
                try:
                    probability_by_frame[int(frame_index)] = float(frame_obj[key])
                except (TypeError, ValueError):
                    pass
    return FrameDetections(by_frame=by_frame, probability_by_frame=probability_by_frame)


def _iou(a: BBox, b: BBox) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    inter_w = max(0.0, min(ax2, bx2) - max(ax, bx))
    inter_h = max(0.0, min(ay2, by2) - max(ay, by))
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def _frame_presence(by_frame: Dict[FrameIndex, List[BBox]], frame: FrameIndex) -> bool:
    return bool(by_frame.get(frame, []))


def _calc_iou_stats(
    gt: Dict[FrameIndex, List[BBox]],
    pred: Dict[FrameIndex, List[BBox]],
) -> Tuple[float, float]:
    ious: List[float] = []
    for frame_idx in sorted(gt.keys() | pred.keys()):
        gt_boxes = gt.get(frame_idx, [])
        pred_boxes = pred.get(frame_idx, [])
        if not gt_boxes or not pred_boxes:
            continue
        best_iou = 0.0
        for gt_box in gt_boxes:
            for pred_box in pred_boxes:
                best_iou = max(best_iou, _iou(gt_box, pred_box))
        ious.append(best_iou)

    mean_iou = _safe_div(sum(ious), len(ious)) if ious else 0.0
    iou_50_rate = _safe_div(sum(1 for v in ious if v >= 0.5), len(ious)) if ious else 0.0
    return mean_iou, iou_50_rate


def _first_last_presence(by_frame: Dict[FrameIndex, List[BBox]]) -> Tuple[Optional[int], Optional[int]]:
    present_frames = [idx for idx, boxes in by_frame.items() if boxes]
    if not present_frames:
        return None, None
    return min(present_frames), max(present_frames)


def _calc_latency(
    gt: Dict[FrameIndex, List[BBox]],
    pred: Dict[FrameIndex, List[BBox]],
    entry_tolerance: int,
    exit_tolerance: int,
) -> Tuple[Optional[bool], Optional[bool], Optional[int], Optional[int]]:
    gt_start, gt_end = _first_last_presence(gt)
    pred_start, pred_end = _first_last_presence(pred)

    if gt_start is None:
        return None, None, None, None

    entry_detected = (
        pred_start is not None and abs(int(pred_start) - int(gt_start)) <= entry_tolerance
    )
    exit_detected = (
        pred_end is not None and abs(int(pred_end) - int(gt_end)) <= exit_tolerance
    ) if gt_end is not None else None

    entry_latency = int(pred_start - gt_start) if pred_start is not None else None
    exit_latency = int(pred_end - gt_end) if (pred_end is not None and gt_end is not None) else None

    return entry_detected, exit_detected, entry_latency, exit_latency


def _calc_continuity(
    gt: Dict[FrameIndex, List[BBox]],
    pred: Dict[FrameIndex, List[BBox]],
) -> Tuple[Optional[int], Optional[float]]:
    # Continuity = how long the detector stays "on" while GT remains present.
    frames = sorted(set(gt.keys()) | set(pred.keys()))
    if not frames:
        return None, None

    max_gt_run = 0
    max_correct_run = 0
    current_gt_run = 0
    current_correct_run = 0

    for frame_idx in frames:
        gt_present = _frame_presence(gt, frame_idx)
        pred_present = _frame_presence(pred, frame_idx)

        if gt_present:
            current_gt_run += 1
            max_gt_run = max(max_gt_run, current_gt_run)
        else:
            current_gt_run = 0

        if gt_present and pred_present:
            current_correct_run += 1
            max_correct_run = max(max_correct_run, current_correct_run)
        else:
            current_correct_run = 0

    if max_gt_run == 0:
        return 0, None

    continuity_rate = float(max_correct_run) / float(max_gt_run)
    return max_correct_run, continuity_rate


def evaluate_video(
    *,
    video_name: str,
    gt: FrameDetections,
    pred: FrameDetections,
    fps: Optional[float],
    entry_tolerance: int,
    exit_tolerance: int,
) -> VideoMetrics:
    frames = sorted(set(gt.by_frame.keys()) | set(pred.by_frame.keys()))
    tp = fp = fn = tn = 0
    for frame_idx in frames:
        gt_present = _frame_presence(gt.by_frame, frame_idx)
        pred_present = _frame_presence(pred.by_frame, frame_idx)
        if gt_present and pred_present:
            tp += 1
        elif gt_present and not pred_present:
            fn += 1
        elif not gt_present and pred_present:
            fp += 1
        else:
            tn += 1

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    presence_accuracy = _safe_div(tp + tn, tp + fp + fn + tn)
    presence_iou = _safe_div(tp, tp + fp + fn)

    continuity_max_frames, continuity_rate = _calc_continuity(gt.by_frame, pred.by_frame)

    mean_iou, iou_50_rate = _calc_iou_stats(gt.by_frame, pred.by_frame)

    entry_detected, exit_detected, entry_latency, exit_latency = _calc_latency(
        gt.by_frame,
        pred.by_frame,
        entry_tolerance,
        exit_tolerance,
    )

    false_alarms_per_min = None
    if fps and fps > 0:
        false_alarms_per_min = float(fp / fps * 60.0)

    # Probability alignment compares the per-frame presence probability against GT presence segments.
    prob_present_vals: List[float] = []
    prob_absent_vals: List[float] = []
    prob_match = []
    if pred.probability_by_frame:
        frames = sorted(set(gt.by_frame.keys()) | set(pred.by_frame.keys()) | set(pred.probability_by_frame.keys()))
        for frame_idx in frames:
            if frame_idx not in pred.probability_by_frame:
                continue
            p_val = float(pred.probability_by_frame[frame_idx])
            gt_present = _frame_presence(gt.by_frame, frame_idx)
            if gt_present:
                prob_present_vals.append(p_val)
            else:
                prob_absent_vals.append(p_val)
            prob_match.append((p_val >= PROBABILITY_MATCH_THRESHOLD) == gt_present)

    probability_mean_present = _safe_div(sum(prob_present_vals), len(prob_present_vals)) if prob_present_vals else None
    probability_mean_absent = _safe_div(sum(prob_absent_vals), len(prob_absent_vals)) if prob_absent_vals else None
    probability_match_rate = _safe_div(sum(1 for v in prob_match if v), len(prob_match)) if prob_match else None

    return VideoMetrics(
        video_name=video_name,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=precision,
        recall=recall,
        f1=f1,
        presence_accuracy=presence_accuracy,
        presence_iou=presence_iou,
        person_continuity_max_frames=continuity_max_frames,
        person_continuity_rate=continuity_rate,
        mean_iou=mean_iou,
        iou_50_rate=iou_50_rate,
        entry_detected=entry_detected,
        exit_detected=exit_detected,
        entry_latency_frames=entry_latency,
        exit_latency_frames=exit_latency,
        false_alarms_per_min=false_alarms_per_min,
        probability_mean_present=probability_mean_present,
        probability_mean_absent=probability_mean_absent,
        probability_match_rate=probability_match_rate,
    )


def _aggregate(metrics: Iterable[VideoMetrics]) -> Dict[str, Any]:
    metrics_list = list(metrics)
    tp = sum(m.tp for m in metrics_list)
    fp = sum(m.fp for m in metrics_list)
    fn = sum(m.fn for m in metrics_list)
    tn = sum(m.tn for m in metrics_list)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    presence_accuracy = _safe_div(tp + tn, tp + fp + fn + tn)
    presence_iou = _safe_div(tp, tp + fp + fn)

    ious = [m.mean_iou for m in metrics_list if m.mean_iou > 0]
    iou_50 = [m.iou_50_rate for m in metrics_list if m.iou_50_rate > 0]

    continuity_max = [m.person_continuity_max_frames for m in metrics_list if m.person_continuity_max_frames is not None]
    continuity_rate = [m.person_continuity_rate for m in metrics_list if m.person_continuity_rate is not None]

    prob_present = [m.probability_mean_present for m in metrics_list if m.probability_mean_present is not None]
    prob_absent = [m.probability_mean_absent for m in metrics_list if m.probability_mean_absent is not None]
    prob_match = [m.probability_match_rate for m in metrics_list if m.probability_match_rate is not None]

    entry_latencies = [m.entry_latency_frames for m in metrics_list if m.entry_latency_frames is not None]
    exit_latencies = [m.exit_latency_frames for m in metrics_list if m.exit_latency_frames is not None]

    entry_detected = [m.entry_detected for m in metrics_list if m.entry_detected is not None]
    exit_detected = [m.exit_detected for m in metrics_list if m.exit_detected is not None]

    false_alarms = [m.false_alarms_per_min for m in metrics_list if m.false_alarms_per_min is not None]

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "presence_accuracy": presence_accuracy,
        "presence_iou": presence_iou,
        "mean_iou": _safe_div(sum(ious), len(ious)) if ious else 0.0,
        "iou_50_rate": _safe_div(sum(iou_50), len(iou_50)) if iou_50 else 0.0,
        "person_continuity_max_frames_mean": _safe_div(sum(continuity_max), len(continuity_max)) if continuity_max else None,
        "person_continuity_rate_mean": _safe_div(sum(continuity_rate), len(continuity_rate)) if continuity_rate else None,
        "entry_detected_rate": _safe_div(sum(1 for v in entry_detected if v), len(entry_detected)) if entry_detected else 0.0,
        "exit_detected_rate": _safe_div(sum(1 for v in exit_detected if v), len(exit_detected)) if exit_detected else 0.0,
        "entry_latency_frames_mean": _safe_div(sum(entry_latencies), len(entry_latencies)) if entry_latencies else None,
        "exit_latency_frames_mean": _safe_div(sum(exit_latencies), len(exit_latencies)) if exit_latencies else None,
        "false_alarms_per_min_mean": _safe_div(sum(false_alarms), len(false_alarms)) if false_alarms else None,
        "probability_mean_present_mean": _safe_div(sum(prob_present), len(prob_present)) if prob_present else None,
        "probability_mean_absent_mean": _safe_div(sum(prob_absent), len(prob_absent)) if prob_absent else None,
        "probability_match_rate_mean": _safe_div(sum(prob_match), len(prob_match)) if prob_match else None,
    }


def _load_split(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "test" in payload:
        return payload
    if "split" in payload and "test" in payload["split"]:
        return payload["split"]
    raise ValueError("Split JSON does not contain a test split.")


def _video_stem(path_str: str) -> str:
    return Path(path_str).stem


def _extract_set_name(video_path: Path) -> Optional[str]:
    for part in video_path.parts:
        if SET_NAME_PATTERN.match(part):
            return part
    return None


def _resolve_annotation_path(
    *,
    video_path: Path,
    split_annotation_path: Path,
    annotation_root: Optional[Path],
) -> Optional[Path]:
    if annotation_root is None:
        return split_annotation_path if split_annotation_path.exists() else None

    set_name = _extract_set_name(video_path)
    if set_name is None:
        return split_annotation_path if split_annotation_path.exists() else None

    candidate = annotation_root / set_name / f"{video_path.stem}.json"
    if candidate.exists():
        return candidate

    return split_annotation_path if split_annotation_path.exists() else None


def evaluate_split(
    *,
    split_path: Path,
    predictions_dir: Path,
    annotation_root: Optional[Path],
    fps: Optional[float],
    entry_tolerance: int,
    exit_tolerance: int,
) -> Dict[str, Any]:
    split_payload = _load_split(split_path)
    test_items = split_payload["test"]
    results: List[VideoMetrics] = []
    for item in test_items:
        video_path = Path(item["video"] if "video" in item else item["video_path"])
        annotation_path = Path(item["annotation"] if "annotation" in item else item.get("annotation_path", ""))
        resolved_annotation = _resolve_annotation_path(
            video_path=video_path,
            split_annotation_path=annotation_path,
            annotation_root=annotation_root,
        )
        if resolved_annotation is None:
            continue
        pred_path = predictions_dir / f"{_video_stem(str(video_path))}.json"
        if not pred_path.exists():
            matches = list(predictions_dir.rglob(f"{_video_stem(str(video_path))}.json"))
            if matches:
                pred_path = matches[0]
            else:
                continue
        gt = load_detections(resolved_annotation, class_filter=None)
        pred = load_detections(pred_path)
        metrics = evaluate_video(
            video_name=video_path.name,
            gt=gt,
            pred=pred,
            fps=fps,
            entry_tolerance=entry_tolerance,
            exit_tolerance=exit_tolerance,
        )
        results.append(metrics)

    return {
        "videos": [m.__dict__ for m in results],
        "aggregate": _aggregate(results),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate predictions against WiseNET manual annotations.")
    parser.add_argument("--gt", type=Path, help="Ground-truth JSON path (single video).")
    parser.add_argument("--pred", type=Path, help="Prediction JSON path (single video).")
    parser.add_argument("--split", type=Path, help="Split JSON path (evaluate test split).")
    parser.add_argument("--predictions-dir", type=Path, help="Directory containing predictions JSON files.")
    parser.add_argument(
        "--annotation-root",
        type=Path,
        default=None,
        help="Override annotation root (e.g., automatic_annotations/.../HS_BS).",
    )
    parser.add_argument("--fps", type=float, default=None, help="Frames per second for false alarm rate.")
    parser.add_argument("--entry-tolerance", type=int, default=0, help="Tolerance in frames for entry detection.")
    parser.add_argument("--exit-tolerance", type=int, default=0, help="Tolerance in frames for exit detection.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs") / "project" / "wisenet_metrics.json",
        help="Output metrics JSON.",
    )

    args = parser.parse_args()
    if args.split is not None:
        if args.predictions_dir is None:
            raise SystemExit("--predictions-dir is required when using --split")
        results = evaluate_split(
            split_path=args.split,
            predictions_dir=args.predictions_dir,
            annotation_root=args.annotation_root,
            fps=args.fps,
            entry_tolerance=args.entry_tolerance,
            exit_tolerance=args.exit_tolerance,
        )
    else:
        if args.gt is None or args.pred is None:
            raise SystemExit("--gt and --pred are required for single-video evaluation")
        gt = load_detections(args.gt, class_filter=None)
        pred = load_detections(args.pred)
        metrics = evaluate_video(
            video_name=args.gt.stem,
            gt=gt,
            pred=pred,
            fps=args.fps,
            entry_tolerance=args.entry_tolerance,
            exit_tolerance=args.exit_tolerance,
        )
        results = {"videos": [metrics.__dict__], "aggregate": _aggregate([metrics])}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
