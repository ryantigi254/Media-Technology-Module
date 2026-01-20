from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal, Qt, QTimer, QRect, QObject, QEvent
from PySide6.QtGui import QAction, QImage, QPixmap, QPainter, QPen, QColor
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QSpinBox,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QMessageBox,
    QGroupBox,
    QFormLayout,
    QProgressBar,
    QTextEdit,
    QComboBox,
)

from ssc.config import MotionConfig, RecorderConfig, HOG_TUNING_DEFAULTS
from ssc.pipeline import process_video

DETECTOR_GRACE_FRAMES = 3
DNN_DEFAULT_SEARCH_ROOT = Path(__file__).resolve().parents[2]
ALERT_PROB_THRESHOLD = 0.5
ALERT_ACTIVE_TEXT = "ALERT (GUI): Person presence likely"
ALERT_IDLE_TEXT = "Monitoring (GUI): no active alert"
ALERT_ACTIVE_STYLE = "color: #ff6b6b; font-weight: bold; font-size: 14px;"
ALERT_IDLE_STYLE = "color: #f6c453; font-weight: bold; font-size: 14px;"
AUTO_ROI_MARGIN_RATIO = 0.1

# DNN auto-fill + motion-gated boxes were AI-assisted (GPT-5.2 series) and then edited by us.
# The alert banner wiring and thresholds were AI-assisted to speed up UI glue code.
# Reason: this is non-core UI plumbing; the logic and thresholds were reviewed manually.


def _find_default_dnn_model_path() -> Path | None:
    from ssc.features.dnn_person_detector import find_default_dnn_model_path

    return find_default_dnn_model_path(DNN_DEFAULT_SEARCH_ROOT)


class _HoverStatusFilter(QObject):
    def __init__(self, window: QMainWindow):
        super().__init__(window)
        self._window = window
        self._default_text = "Ready"

    def eventFilter(self, obj, event):
        try:
            if event.type() == QEvent.Enter:
                tip = ""
                if hasattr(obj, "statusTip"):
                    tip = str(obj.statusTip() or "")
                if not tip and hasattr(obj, "toolTip"):
                    tip = str(obj.toolTip() or "")
                if tip:
                    self._window.statusBar().showMessage(tip)
            elif event.type() == QEvent.Leave:
                self._window.statusBar().showMessage(self._default_text)
        except KeyboardInterrupt:
            return False
        except Exception:
            return False
        return super().eventFilter(obj, event)


@dataclass
class GuiState:
    input_path: Path | None = None
    output_dir: Path = Path("outputs")
    roi_rects: list[tuple[int, int, int, int]] | None = None
    roi_mode: str = "manual"
    motion_cfg: MotionConfig | None = None
    recorder_cfg: RecorderConfig | None = None
    person_detector: str = "hog"
    dnn_model_path: Path | None = None


class VideoThread(QThread):
    change_pixmap_signal = Signal(QImage)
    frame_ready = Signal(np.ndarray, int)

    def __init__(self, path: Path):
        super().__init__()
        self.path = path
        self.running = True
        self.cap = cv2.VideoCapture(str(path))
        self.frame_idx = -1

    def run(self):
        while self.running and self.cap.isOpened():
            ok, frame = self.cap.read()
            if not ok:
                break
            self.frame_idx += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.change_pixmap_signal.emit(qt_image)
            self.frame_ready.emit(frame, self.frame_idx)
            self.msleep(33)  # ~30 fps

    def stop(self):
        self.running = False
        self.wait()


class DetectionThread(QThread):
    finished = Signal(int)
    status_update = Signal(str)
    frame_annotated = Signal(QImage, int, object)  # frame, frame_idx, presence_p (float or None)

    def __init__(
        self,
        input_path: Path,
        output_dir: Path,
        motion_cfg: MotionConfig,
        recorder_cfg: RecorderConfig,
        stop_event: threading.Event,
        debug: bool,
        person_detector: str,
        dnn_model_path: Path | None,
    ):
        super().__init__()
        self.input_path = input_path
        self.output_dir = output_dir
        self.motion_cfg = motion_cfg
        self.recorder_cfg = recorder_cfg
        self.stop_event = stop_event
        self.debug = bool(debug)
        self.person_detector = str(person_detector or "hog")
        self.dnn_model_path = dnn_model_path

    def run(self):
        try:
            # Override process_video to emit frames for live preview
            from ssc.video_io import open_capture, get_video_info
            from ssc.motion import MotionDetector
            from ssc.recorder import IncidentRecorder
            from ssc.features.probabilistic_presence import ProbabilisticPresenceTracker
            from ssc.features.dnn_person_detector import detect_people_dnn
            from ssc.features.yolo_person_detector import detect_objects_yolo, is_yolo_available
            from ssc.pipeline import _annotate, _make_run_dir
            import json
            from datetime import datetime
            from pathlib import Path

            cap = open_capture(self.input_path)
            info = get_video_info(cap)

            run_dir = _make_run_dir(base_output_dir=self.output_dir, input_path=self.input_path)
            run_dir.mkdir(parents=True, exist_ok=True)

            run_meta = {
                "input_path": str(self.input_path),
                "output_dir": str(run_dir),
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "video_info": {
                    "fps": info.fps,
                    "width": info.width,
                    "height": info.height,
                    "frame_count": info.frame_count,
                },
                "motion_cfg": self.motion_cfg.__dict__,
                "recorder_cfg": self.recorder_cfg.__dict__,
            }
            (run_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

            detector = MotionDetector(self.motion_cfg)
            recorder = IncidentRecorder(output_dir=run_dir, video_info=info, cfg=self.recorder_cfg)
            prob_tracker = ProbabilisticPresenceTracker() if (self.motion_cfg.enable_smoothing or self.motion_cfg.show_confidence) else None

            detector_mode = (self.person_detector or "hog").strip().lower()
            if detector_mode not in ("hog", "dnn", "yolo", "motion"):
                detector_mode = "hog"

            if detector_mode == "dnn":
                model_path = self.dnn_model_path
                if (
                    model_path is None
                    or not model_path.is_file()
                    or not model_path.with_suffix(".prototxt").is_file()
                    or not model_path.with_suffix(".caffemodel").is_file()
                ):
                    self.status_update.emit(
                        "DNN model not found/invalid; falling back to motion-only boxes."
                    )
                    detector_mode = "motion"

            if detector_mode == "yolo" and not is_yolo_available():
                self.status_update.emit(
                    "Ultralytics YOLO not installed; falling back to motion-only boxes."
                )
                detector_mode = "motion"

            hog = None
            if detector_mode == "hog":
                hog = cv2.HOGDescriptor()
                hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

            person_interval = 3 if detector_mode == "hog" else 5
            last_person_frame_idx = -10_000
            last_person_bboxes: list[tuple[int, int, int, int]] = []
            last_person_labels: list[str] = []
            last_person_positive_idx = -10_000

            # Project note:
            # We use person detectors for more full-body style boxes than raw motion blobs.
            # We only run them when motion is stable to keep performance reasonable.

            roi_area = 0
            for x1, y1, x2, y2 in self.motion_cfg.roi_rects:
                roi_area += max(0, int(x2) - int(x1)) * max(0, int(y2) - int(y1))
            mask_den = float(roi_area) if roi_area > 0 else float(info.width * info.height)

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

            def _filter_to_roi(bboxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
                rois = list(self.motion_cfg.roi_rects)
                if not rois:
                    return list(bboxes)
                filtered: list[tuple[int, int, int, int]] = []
                for x, y, w, h in bboxes:
                    cx = float(x) + float(w) * 0.5
                    cy = float(y) + float(h) * 0.5
                    for rx1, ry1, rx2, ry2 in rois:
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

            def _filter_to_roi_labeled(
                bboxes: list[tuple[int, int, int, int]],
                labels: list[str],
            ) -> tuple[list[tuple[int, int, int, int]], list[str]]:
                rois = list(self.motion_cfg.roi_rects)
                if not rois:
                    return list(bboxes), list(labels)
                filtered_boxes: list[tuple[int, int, int, int]] = []
                filtered_labels: list[str] = []
                for (x, y, w, h), label in zip(bboxes, labels, strict=False):
                    cx = float(x) + float(w) * 0.5
                    cy = float(y) + float(h) * 0.5
                    for rx1, ry1, rx2, ry2 in rois:
                        if float(rx1) <= cx <= float(rx2) and float(ry1) <= cy <= float(ry2):
                            # Clip box to this ROI
                            nx1 = max(int(x), int(rx1))
                            ny1 = max(int(y), int(ry1))
                            nx2 = min(int(x + w), int(rx2))
                            ny2 = min(int(y + h), int(ry2))
                            
                            if nx2 > nx1 and ny2 > ny1:
                                filtered_boxes.append((nx1, ny1, nx2 - nx1, ny2 - ny1))
                                filtered_labels.append(str(label))
                            break
                return filtered_boxes, filtered_labels

            def _hog_people(frame_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
                if hog is None:
                    return []

                out: list[tuple[int, int, int, int]] = []

                rois = list(self.motion_cfg.roi_rects)
                if not rois:
                    rois = [(0, 0, int(info.width), int(info.height))]

                for x1, y1, x2, y2 in rois:
                    ix1 = max(int(x1), 0)
                    iy1 = max(int(y1), 0)
                    ix2 = min(int(x2), int(info.width))
                    iy2 = min(int(y2), int(info.height))
                    if ix2 <= ix1 or iy2 <= iy1:
                        continue

                    roi = frame_bgr[iy1:iy2, ix1:ix2]
                    if roi.size == 0:
                        continue

                    rh, rw = roi.shape[:2]
                    if rw < 64 or rh < 128:
                        continue

                    scale = 1.0
                    target_w = 480
                    if rw > target_w:
                        scale = float(target_w) / float(rw)
                        roi = cv2.resize(roi, (int(round(rw * scale)), int(round(rh * scale))))

                    rects, weights = hog.detectMultiScale(
                        roi,
                        winStride=(16, 16),
                        padding=(8, 8),
                        scale=1.08,
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
                            out.append((int(fx1), int(fy1), int(fx2 - fx1), int(fy2 - fy1)))

                return out

            def _person_boxes(frame_bgr: np.ndarray) -> tuple[list[tuple[int, int, int, int]], list[str]]:
                if detector_mode == "hog":
                    return _filter_to_roi(_hog_people(frame_bgr)), []
                if detector_mode == "dnn":
                    if self.dnn_model_path is None:
                        return [], []
                    boxes = detect_people_dnn(frame_bgr, self.dnn_model_path)
                    return _filter_to_roi(boxes), ["person"] * len(boxes)
                if detector_mode == "yolo":
                    detections = detect_objects_yolo(frame_bgr, return_labels=True)
                    boxes = [d[0] for d in detections]
                    labels = [d[1] for d in detections]
                    return _filter_to_roi_labeled(boxes, labels)
                return [], []

            frame_idx = -1
            processed = 0
            motion_frames = 0
            stable_motion = False
            on_counter = 0
            off_counter = 0
            cancelled = False
            last_dbg_emit = -10_000

            try:
                while True:
                    if self.stop_event.is_set():
                        cancelled = True
                        break

                    ok, frame = cap.read()
                    if not ok:
                        break

                    frame_idx += 1
                    res = detector.detect(frame)

                    # Suppress global light-change events where the mask "floods" most of the frame.
                    # This prevents the probability/motion state being driven by illumination changes.
                    mask_flood_ratio = float(np.count_nonzero(res.motion_mask)) / mask_den
                    mask_flooded = mask_flood_ratio >= 0.60

                    # Project note:
                    # If the motion mask covers most of the ROI, it often means lighting changed (global illumination).
                    # In that case we suppress updates to avoid false positives / ghost boxes.

                    on_frames = max(int(self.motion_cfg.motion_on_frames), 1)
                    off_frames = max(int(self.motion_cfg.motion_off_frames), 1)

                    raw_motion = bool(res.motion)
                    if not mask_flooded:
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

                    effective_motion = (stable_motion and raw_motion) if not mask_flooded else stable_motion

                    if detector_mode == "yolo":
                        if (frame_idx - last_person_frame_idx) >= person_interval:
                            last_person_bboxes, last_person_labels = _person_boxes(frame)
                            last_person_frame_idx = frame_idx
                            if last_person_bboxes:
                                last_person_positive_idx = frame_idx
                    elif stable_motion and raw_motion and (not mask_flooded) and detector_mode in ("hog", "dnn"):
                        if (frame_idx - last_person_frame_idx) >= person_interval:
                            last_person_bboxes, last_person_labels = _person_boxes(frame)
                            last_person_frame_idx = frame_idx
                            if last_person_bboxes:
                                last_person_positive_idx = frame_idx
                    elif detector_mode != "yolo":
                        last_person_bboxes = []
                        last_person_labels = []

                    if self.debug and (frame_idx - last_dbg_emit) >= 15:
                        bbox_source = "none"
                        if mask_flooded:
                            bbox_source = "flood"
                        elif stable_motion:
                            if detector_mode in ("hog", "dnn", "yolo") and last_person_bboxes:
                                bbox_source = detector_mode
                            else:
                                bbox_source = "motion" if res.bboxes else "none"
                        self.status_update.emit(
                            f"[dbg] frame={frame_idx} stable={int(stable_motion)} flood={int(mask_flooded)} "
                            f"flood_ratio={mask_flood_ratio:.3f} src={bbox_source} person_n={len(last_person_bboxes)} motion_n={len(res.bboxes)}"
                        )
                        last_dbg_emit = frame_idx

                    # --- Selection Logic ---
                    p_val: float | None = None
                    display_bboxes: list[tuple[int, int, int, int]] = []

                    # 1. Update Tracker
                    if prob_tracker is not None:
                        if not mask_flooded:
                            detected_bbox = None
                            # Grace period logic for tracker input
                            if last_person_bboxes:
                                candidate = last_person_bboxes
                            elif stable_motion and raw_motion:
                                candidate = _filter_to_roi(res.bboxes)
                            else:
                                candidate = []

                            if candidate:
                                detected_bbox = candidate[0] if len(candidate) == 1 else _union_xywh(candidate)

                            _ = prob_tracker.update(detected_bbox=detected_bbox)
                        
                        if self.motion_cfg.show_confidence:
                            p_val = prob_tracker.probability()

                    # 2. Decide Display Boxes
                    if self.motion_cfg.enable_smoothing and prob_tracker is not None:
                        if mask_flooded:
                            display_bboxes = []
                        else:
                            draw_motion = prob_tracker.should_draw()
                            draw_bbox = prob_tracker.bbox()
                            display_bboxes = [draw_bbox] if (draw_motion and draw_bbox is not None) else []
                    else:
                        # Raw Boxes
                        if mask_flooded:
                            display_bboxes = []
                        elif detector_mode == "yolo" and last_person_bboxes:
                            display_bboxes = last_person_bboxes
                        elif stable_motion and raw_motion:
                            display_bboxes = last_person_bboxes if last_person_bboxes else _filter_to_roi(res.bboxes)
                        else:
                            display_bboxes = []

                    # 3. Annotate
                    should_draw_motion = bool(display_bboxes) or (stable_motion and raw_motion and not mask_flooded)
                    annotated = _annotate(frame, frame_idx, display_bboxes, should_draw_motion, presence_p=p_val)
                    p = 1.0 if stable_motion else 0.0 # for signal (legacy usage)

                    if self.motion_cfg.show_confidence and p_val is not None:
                         p = p_val

                    if detector_mode == "yolo" and last_person_bboxes and last_person_labels:
                        for (x, y, w, h), label in zip(last_person_bboxes, last_person_labels, strict=False):
                            x1 = max(int(x), 0)
                            y1 = max(int(y) - 6, 0)
                            cv2.putText(
                                annotated,
                                str(label),
                                (x1, y1),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255, 255, 0),
                                2,
                                cv2.LINE_AA,
                            )

                    recorder.update(frame_idx, annotated, motion=effective_motion)

                    if effective_motion:
                        motion_frames += 1

                    # Emit annotated frame for live preview
                    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    self.frame_annotated.emit(qt_image, frame_idx, p)

                    processed += 1
                    if self.stop_event.is_set():
                        cancelled = True
                        break

            finally:
                recorder.close(final_frame_idx=frame_idx)
                cap.release()

            self.finished.emit(0 if not cancelled else 1)
        except Exception as e:
            self.status_update.emit(f"Error: {e}")
            self.finished.emit(1)


class VideoPreviewWidget(QWidget):
    roi_selected = Signal(list)

    def __init__(self):
        super().__init__()
        self.image = QPixmap()
        self.roi_rects_frame: list[tuple[int, int, int, int]] = []
        self.drawing = False
        self.start_pos = None
        self.current_rect = None
        self._image_draw_rect: QRect | None = None
        self.roi_enabled = True
        self.setMinimumSize(640, 480)

    def set_image(self, pixmap: QPixmap):
        self.image = pixmap
        self.update()

    def set_rois_frame(self, rois: list[tuple[int, int, int, int]]):
        self.roi_rects_frame = list(rois)
        self.update()

    def set_roi_enabled(self, enabled: bool):
        self.roi_enabled = bool(enabled)
        if not self.roi_enabled:
            self.drawing = False
            self.current_rect = None
        self.update()

    def _compute_image_draw_rect(self) -> QRect:
        if self.image.isNull():
            return QRect(0, 0, 0, 0)

        ww = int(self.width())
        wh = int(self.height())
        iw = int(self.image.width())
        ih = int(self.image.height())

        if ww <= 0 or wh <= 0 or iw <= 0 or ih <= 0:
            return QRect(0, 0, 0, 0)

        scale = min(float(ww) / float(iw), float(wh) / float(ih))
        dw = int(round(iw * scale))
        dh = int(round(ih * scale))
        dx = int((ww - dw) // 2)
        dy = int((wh - dh) // 2)
        return QRect(dx, dy, dw, dh)

    def get_rois_frame(self) -> list[tuple[int, int, int, int]]:
        return list(self.roi_rects_frame)

    def paintEvent(self, event):
        painter = QPainter(self)
        if self.image.isNull():
            # Dark placeholder with centered text
            painter.fillRect(self.rect(), QColor(48, 48, 48))
            painter.setPen(QPen(QColor(180, 180, 180)))
            font = painter.font()
            font.setPointSize(14)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignCenter, "Open a video to set ROI")
        else:
            draw_rect = self._compute_image_draw_rect()
            self._image_draw_rect = draw_rect
            painter.drawPixmap(draw_rect, self.image, self.image.rect())

        # Green ROI rectangles
        pen = QPen(QColor(46, 204, 113), 2, Qt.SolidLine)
        painter.setPen(pen)

        if not self.image.isNull():
            draw_rect = self._compute_image_draw_rect()
            iw = int(self.image.width())
            ih = int(self.image.height())
            if draw_rect.width() > 0 and draw_rect.height() > 0:
                for fx1, fy1, fx2, fy2 in self.roi_rects_frame:
                    x1 = int(round(draw_rect.left() + (fx1 * draw_rect.width() / iw)))
                    y1 = int(round(draw_rect.top() + (fy1 * draw_rect.height() / ih)))
                    x2 = int(round(draw_rect.left() + (fx2 * draw_rect.width() / iw)))
                    y2 = int(round(draw_rect.top() + (fy2 * draw_rect.height() / ih)))
                    painter.drawRect(x1, y1, x2 - x1, y2 - y1)

        if self.current_rect:
            painter.drawRect(*self.current_rect)

    def mousePressEvent(self, event):
        if not self.roi_enabled:
            return
        if event.button() == Qt.LeftButton:
            if self.image.isNull():
                return

            img_rect = self._compute_image_draw_rect()
            if not img_rect.contains(event.pos()):
                return
            self.drawing = True
            self.start_pos = event.pos()

    def mouseMoveEvent(self, event):
        if not self.roi_enabled:
            return
        if self.drawing and self.start_pos:
            img_rect = self._compute_image_draw_rect()
            max_x = img_rect.left() + img_rect.width()
            max_y = img_rect.top() + img_rect.height()
            ex = min(max(event.pos().x(), img_rect.left()), max_x)
            ey = min(max(event.pos().y(), img_rect.top()), max_y)
            self.current_rect = (
                self.start_pos.x(),
                self.start_pos.y(),
                ex - self.start_pos.x(),
                ey - self.start_pos.y(),
            )
            self.update()

    def mouseReleaseEvent(self, event):
        if not self.roi_enabled:
            return
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            if self.start_pos:
                img_rect = self._compute_image_draw_rect()
                max_x = img_rect.left() + img_rect.width()
                max_y = img_rect.top() + img_rect.height()
                x1, y1 = self.start_pos.x(), self.start_pos.y()
                x2 = min(max(event.pos().x(), img_rect.left()), max_x)
                y2 = min(max(event.pos().y(), img_rect.top()), max_y)

                if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                    # Normalize to (x1,y1,x2,y2) with x2>x1, y2>y1
                    rx1, rx2 = sorted((x1, x2))
                    ry1, ry2 = sorted((y1, y2))
                    draw_rect = img_rect
                    iw = int(self.image.width())
                    ih = int(self.image.height())
                    fx1 = int(round((rx1 - draw_rect.left()) * iw / draw_rect.width()))
                    fy1 = int(round((ry1 - draw_rect.top()) * ih / draw_rect.height()))
                    fx2 = int(round((rx2 - draw_rect.left()) * iw / draw_rect.width()))
                    fy2 = int(round((ry2 - draw_rect.top()) * ih / draw_rect.height()))

                    fx1 = max(0, min(iw - 1, fx1))
                    fy1 = max(0, min(ih - 1, fy1))
                    fx2 = max(0, min(iw, fx2))
                    fy2 = max(0, min(ih, fy2))

                    if fx2 > fx1 and fy2 > fy1:
                        self.roi_rects_frame.append((int(fx1), int(fy1), int(fx2), int(fy2)))
            self.start_pos = None
            self.current_rect = None
            self.update()
            self.roi_selected.emit(self.roi_rects_frame)

    def clear_rois(self):
        self.roi_rects_frame.clear()
        self.update()

    def get_rois(self) -> list[tuple[int, int, int, int]]:
        return list(self.roi_rects_frame)


class SmartSecurityCameraGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.state = GuiState()
        self.video_thread: VideoThread | None = None
        self.detection_thread: DetectionThread | None = None
        self.stop_event = threading.Event()
        self._hover_filter: _HoverStatusFilter | None = None
        self.init_ui()
        self.apply_defaults()

    def init_ui(self):
        self.setWindowTitle("Smart Security Camera (Qt)")
        self.setGeometry(100, 100, 1200, 800)
        self.statusBar().showMessage("Ready")
        self._hover_filter = _HoverStatusFilter(self)

        # Modern dark theme stylesheet
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #f0f0f0;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 10px;
                background-color: #333;
                color: #f0f0f0;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QLabel {
                color: #f0f0f0;
            }
            QPushButton {
                background-color: #4a90e2;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
            QPushButton:pressed {
                background-color: #2968a3;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #888;
            }
            QSpinBox, QDoubleSpinBox, QCheckBox {
                background-color: #3c3c3c;
                color: #f0f0f0;
                border: 1px solid #555;
                padding: 4px;
                border-radius: 4px;
            }
            QProgressBar {
                border: 1px solid #555;
                border-radius: 4px;
                text-align: center;
                background-color: #3c3c3c;
            }
            QProgressBar::chunk {
                background-color: #4a90e2;
                border-radius: 3px;
            }
            QTextEdit {
                background-color: #3c3c3c;
                color: #f0f0f0;
                border: 1px solid #555;
                border-radius: 4px;
            }
            QToolTip {
                background-color: #555;
                color: #f0f0f0;
                border: 1px solid #777;
                padding: 4px;
                border-radius: 4px;
                font-size: 12px;
            }
        """)
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # Left: preview + ROI
        left_layout = QVBoxLayout()
        left_layout.setSpacing(12)

        # Video selector
        video_group = QGroupBox("Video Input")
        video_layout = QVBoxLayout()
        self.open_video_btn = QPushButton("Open Video")
        self.open_video_btn.clicked.connect(self.open_video)
        self.open_video_btn.setMinimumHeight(40)
        self.open_video_btn.setStatusTip("Open a video file to preview the first frames and draw ROI(s).")
        self.open_video_btn.installEventFilter(self._hover_filter)
        video_layout.addWidget(self.open_video_btn)
        video_group.setLayout(video_layout)
        left_layout.addWidget(video_group)

        # Preview
        preview_group = QGroupBox("Preview & ROI")
        preview_layout = QVBoxLayout()
        self.preview = VideoPreviewWidget()
        self.preview.roi_selected.connect(self.on_roi_selected)
        preview_layout.addWidget(self.preview)

        roi_mode_layout = QHBoxLayout()
        roi_mode_tip = "Choose how ROI is set: draw rectangles manually or use the full frame automatically."
        roi_mode_label = QLabel("ROI Mode")
        roi_mode_label.setToolTip(roi_mode_tip)
        roi_mode_label.setStatusTip(roi_mode_tip)
        roi_mode_label.installEventFilter(self._hover_filter)
        self.roi_mode_combo = QComboBox()
        self.roi_mode_combo.addItem("Draw ROI (manual)", "manual")
        self.roi_mode_combo.addItem("Automatic (full frame)", "auto")
        self.roi_mode_combo.setToolTip(roi_mode_tip)
        self.roi_mode_combo.setStatusTip(roi_mode_tip)
        self.roi_mode_combo.installEventFilter(self._hover_filter)
        self.roi_mode_combo.currentIndexChanged.connect(self.on_roi_mode_changed)
        roi_mode_layout.addWidget(roi_mode_label)
        roi_mode_layout.addWidget(self.roi_mode_combo)
        roi_mode_layout.addStretch()
        preview_layout.addLayout(roi_mode_layout)

        roi_controls = QHBoxLayout()
        self.clear_roi_btn = QPushButton("Clear ROIs")
        self.clear_roi_btn.clicked.connect(self.clear_rois)
        self.clear_roi_btn.setEnabled(False)
        self.clear_roi_btn.setStatusTip("Clear all selected ROI rectangles.")
        self.clear_roi_btn.installEventFilter(self._hover_filter)
        roi_controls.addWidget(self.clear_roi_btn)
        roi_controls.addStretch()
        preview_layout.addLayout(roi_controls)

        preview_group.setLayout(preview_layout)
        left_layout.addWidget(preview_group)

        layout.addLayout(left_layout, 2)

        # Right: settings + controls
        right_layout = QVBoxLayout()
        settings_group = QGroupBox("Motion Detection Settings")
        form = QFormLayout()
        self._hog_setting_rows: list[tuple[QLabel, QWidget]] = []

        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(50, 50000)
        min_area_tip = "Minimum contour area (pixels) to count as motion. Lower = more sensitive; higher = less noise."
        self.min_area_spin.setToolTip(min_area_tip)
        self.min_area_spin.setStatusTip(min_area_tip)
        min_area_label = QLabel("Min Area")
        min_area_label.setToolTip(min_area_tip)
        min_area_label.setStatusTip(min_area_tip)
        min_area_label.installEventFilter(self._hover_filter)
        self.min_area_spin.installEventFilter(self._hover_filter)
        form.addRow(min_area_label, self.min_area_spin)
        self._hog_setting_rows.append((min_area_label, self.min_area_spin))

        self.post_frames_spin = QSpinBox()
        self.post_frames_spin.setRange(0, 500)
        post_frames_tip = "Keep recording this many frames after motion stops (extends incident clips)."
        self.post_frames_spin.setToolTip(post_frames_tip)
        self.post_frames_spin.setStatusTip(post_frames_tip)
        post_frames_label = QLabel("Post Frames")
        post_frames_label.setToolTip(post_frames_tip)
        post_frames_label.setStatusTip(post_frames_tip)
        post_frames_label.installEventFilter(self._hover_filter)
        self.post_frames_spin.installEventFilter(self._hover_filter)
        form.addRow(post_frames_label, self.post_frames_spin)

        self.close_iters_spin = QSpinBox()
        self.close_iters_spin.setRange(0, 10)
        close_tip = "Morphological close iterations (fills gaps in motion mask)."
        self.close_iters_spin.setToolTip(close_tip)
        self.close_iters_spin.setStatusTip(close_tip)
        close_label = QLabel("Close Iters")
        close_label.setToolTip(close_tip)
        close_label.setStatusTip(close_tip)
        close_label.installEventFilter(self._hover_filter)
        self.close_iters_spin.installEventFilter(self._hover_filter)
        form.addRow(close_label, self.close_iters_spin)
        self._hog_setting_rows.append((close_label, self.close_iters_spin))

        self.dilate_iters_spin = QSpinBox()
        self.dilate_iters_spin.setRange(0, 10)
        dilate_tip = "Dilation iterations (connects nearby blobs; lower = more separate people)."
        self.dilate_iters_spin.setToolTip(dilate_tip)
        self.dilate_iters_spin.setStatusTip(dilate_tip)
        dilate_label = QLabel("Dilate Iters")
        dilate_label.setToolTip(dilate_tip)
        dilate_label.setStatusTip(dilate_tip)
        dilate_label.installEventFilter(self._hover_filter)
        self.dilate_iters_spin.installEventFilter(self._hover_filter)
        form.addRow(dilate_label, self.dilate_iters_spin)
        self._hog_setting_rows.append((dilate_label, self.dilate_iters_spin))

        self.merge_pad_spin = QSpinBox()
        self.merge_pad_spin.setRange(0, 100)
        merge_tip = "Padding when merging nearby boxes (helps merge limbs)."
        self.merge_pad_spin.setToolTip(merge_tip)
        self.merge_pad_spin.setStatusTip(merge_tip)
        merge_label = QLabel("Merge Pad")
        merge_label.setToolTip(merge_tip)
        merge_label.setStatusTip(merge_tip)
        merge_label.installEventFilter(self._hover_filter)
        self.merge_pad_spin.installEventFilter(self._hover_filter)
        form.addRow(merge_label, self.merge_pad_spin)
        self._hog_setting_rows.append((merge_label, self.merge_pad_spin))

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(-1.0, 1.0)
        self.lr_spin.setSingleStep(0.01)
        self.lr_spin.setDecimals(3)  # allow 0.005
        lr_tip = "MOG2 learningRate. -1=auto, 0=no update, small values update slowly (affects background adaptation)."
        self.lr_spin.setToolTip(lr_tip)
        self.lr_spin.setStatusTip(lr_tip)
        lr_label = QLabel("Learning Rate")
        lr_label.setToolTip(lr_tip)
        lr_label.setStatusTip(lr_tip)
        lr_label.installEventFilter(self._hover_filter)
        self.lr_spin.installEventFilter(self._hover_filter)
        form.addRow(lr_label, self.lr_spin)
        self._hog_setting_rows.append((lr_label, self.lr_spin))

        self.on_frames_spin = QSpinBox()
        self.on_frames_spin.setRange(1, 30)
        on_tip = "Require this many consecutive motion frames before triggering (reduces 1-frame ghosts)."
        self.on_frames_spin.setToolTip(on_tip)
        self.on_frames_spin.setStatusTip(on_tip)
        on_label = QLabel("On Frames")
        on_label.setToolTip(on_tip)
        on_label.setStatusTip(on_tip)
        on_label.installEventFilter(self._hover_filter)
        self.on_frames_spin.installEventFilter(self._hover_filter)
        form.addRow(on_label, self.on_frames_spin)
        self._hog_setting_rows.append((on_label, self.on_frames_spin))

        self.off_frames_spin = QSpinBox()
        self.off_frames_spin.setRange(1, 60)
        off_tip = "Require this many consecutive no-motion frames before clearing motion (reduces flicker)."
        self.off_frames_spin.setToolTip(off_tip)
        self.off_frames_spin.setStatusTip(off_tip)
        off_label = QLabel("Off Frames")
        off_label.setToolTip(off_tip)
        off_label.setStatusTip(off_tip)
        off_label.installEventFilter(self._hover_filter)
        self.off_frames_spin.installEventFilter(self._hover_filter)
        form.addRow(off_label, self.off_frames_spin)
        self._hog_setting_rows.append((off_label, self.off_frames_spin))

        self.person_detector_combo = QComboBox()
        detector_tip = "Select person detector used to refine boxes inside the ROI."
        self.person_detector_combo.addItem("HOG (fast)", "hog")
        self.person_detector_combo.addItem("DNN (MobileNet-SSD)", "dnn")
        self.person_detector_combo.addItem("YOLO (Ultralytics)", "yolo")
        self.person_detector_combo.addItem("Motion only", "motion")
        self.person_detector_combo.setToolTip(detector_tip)
        self.person_detector_combo.setStatusTip(detector_tip)
        detector_label = QLabel("Person Detector")
        detector_label.setToolTip(detector_tip)
        detector_label.setStatusTip(detector_tip)
        detector_label.installEventFilter(self._hover_filter)
        self.person_detector_combo.installEventFilter(self._hover_filter)
        self.person_detector_combo.currentIndexChanged.connect(self.on_detector_changed)
        form.addRow(detector_label, self.person_detector_combo)

        self.single_box_cb = QCheckBox()
        single_tip = "Force a single merged bbox per frame (useful when there is one person)."
        self.single_box_cb.setToolTip(single_tip)
        self.single_box_cb.setStatusTip(single_tip)
        single_label = QLabel("Single Box")
        single_label.setToolTip(single_tip)
        single_label.setStatusTip(single_tip)
        single_label.installEventFilter(self._hover_filter)
        self.single_box_cb.installEventFilter(self._hover_filter)
        form.addRow(single_label, self.single_box_cb)
        self._hog_setting_rows.append((single_label, self.single_box_cb))

        self.cc_cb = QCheckBox()
        cc_tip = "Use connected components for blob extraction (alternative to contours)."
        self.cc_cb.setToolTip(cc_tip)
        self.cc_cb.setStatusTip(cc_tip)
        cc_label = QLabel("Connected Components")
        cc_label.setToolTip(cc_tip)
        cc_label.setStatusTip(cc_tip)
        cc_label.installEventFilter(self._hover_filter)
        self.cc_cb.installEventFilter(self._hover_filter)
        form.addRow(cc_label, self.cc_cb)
        self._hog_setting_rows.append((cc_label, self.cc_cb))

        self.prob_smooth_cb = QCheckBox()
        smooth_tip = "Enable probabilistic presence smoothing (EMA logic) for stable boxes."
        self.prob_smooth_cb.setToolTip(smooth_tip)
        self.prob_smooth_cb.setStatusTip(smooth_tip)
        smooth_label = QLabel("Enable Smoothing")
        smooth_label.setToolTip(smooth_tip)
        smooth_label.setStatusTip(smooth_tip)
        smooth_label.installEventFilter(self._hover_filter)
        self.prob_smooth_cb.installEventFilter(self._hover_filter)
        form.addRow(smooth_label, self.prob_smooth_cb)

        self.prob_conf_cb = QCheckBox()
        conf_tip = "Show the presence probability score (e.g. P=0.95) on the video."
        self.prob_conf_cb.setToolTip(conf_tip)
        self.prob_conf_cb.setStatusTip(conf_tip)
        conf_label = QLabel("Show Confidence")
        conf_label.setToolTip(conf_tip)
        conf_label.setStatusTip(conf_tip)
        conf_label.installEventFilter(self._hover_filter)
        self.prob_conf_cb.installEventFilter(self._hover_filter)
        form.addRow(conf_label, self.prob_conf_cb)

        self.debug_cb = QCheckBox()
        debug_tip = "Show debug lines in Status: stable motion, light-change flood suppression, and whether HOG/DNN or motion boxes are used."
        self.debug_cb.setToolTip(debug_tip)
        self.debug_cb.setStatusTip(debug_tip)
        debug_label = QLabel("Debug")
        debug_label.setToolTip(debug_tip)
        debug_label.setStatusTip(debug_tip)
        debug_label.installEventFilter(self._hover_filter)
        self.debug_cb.installEventFilter(self._hover_filter)
        form.addRow(debug_label, self.debug_cb)

        settings_group.setLayout(form)
        right_layout.addWidget(settings_group)

        # Output folder
        out_group = QGroupBox("Output")
        out_layout = QVBoxLayout()
        self.output_label = QLabel("outputs")
        self.output_label.setStatusTip("Base output folder. Runs are stored under outputs/<set>/<video>/run_<timestamp>.")
        self.output_label.installEventFilter(self._hover_filter)
        out_layout.addWidget(self.output_label)
        self.browse_out_btn = QPushButton("Browse")
        self.browse_out_btn.clicked.connect(self.browse_output)
        self.browse_out_btn.setStatusTip("Select the base output folder for new runs.")
        self.browse_out_btn.installEventFilter(self._hover_filter)
        out_layout.addWidget(self.browse_out_btn)
        out_group.setLayout(out_layout)
        right_layout.addWidget(out_group)

        # Controls
        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout()
        self.run_btn = QPushButton("Run Detection")
        self.run_btn.clicked.connect(self.run_detection)
        self.run_btn.setEnabled(False)
        self.run_btn.setStatusTip("Start detection using the current settings and ROI(s).")
        self.run_btn.installEventFilter(self._hover_filter)
        controls_layout.addWidget(self.run_btn)

        self.stop_btn = QPushButton("Stop / Cancel")
        self.stop_btn.clicked.connect(self.stop_detection)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStatusTip("Stop processing early (cancels the current run).")
        self.stop_btn.installEventFilter(self._hover_filter)
        controls_layout.addWidget(self.stop_btn)

        self.open_out_btn = QPushButton("Open Outputs Folder")
        self.open_out_btn.clicked.connect(self.open_outputs)
        self.open_out_btn.setStatusTip("Open the base output folder in your file explorer.")
        self.open_out_btn.installEventFilter(self._hover_filter)
        controls_layout.addWidget(self.open_out_btn)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        controls_layout.addWidget(self.progress)

        controls_group.setLayout(controls_layout)
        right_layout.addWidget(controls_group)

        # Status
        # GUI alert banner (local-only): for deployment, prefer a web-service/dashboard to push email/SMS/webhook alerts.
        self.alert_label = QLabel(ALERT_IDLE_TEXT)
        self.alert_label.setAlignment(Qt.AlignCenter)
        self.alert_label.setStyleSheet(ALERT_IDLE_STYLE)
        self.alert_label.setToolTip(
            "Local GUI alert. For deployment, use a web service/dashboard for live feed updates and email/SMS/webhook alerts."
        )
        right_layout.addWidget(self.alert_label)

        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(120)
        right_layout.addWidget(QLabel("Status"))
        right_layout.addWidget(self.status_text)

        right_layout.addStretch()
        layout.addLayout(right_layout, 1)

    def apply_defaults(self):
        # Apply defaults from your image
        self.min_area_spin.setValue(HOG_TUNING_DEFAULTS.min_contour_area)
        self.post_frames_spin.setValue(30)
        self.close_iters_spin.setValue(HOG_TUNING_DEFAULTS.morph_close_iterations)
        self.dilate_iters_spin.setValue(HOG_TUNING_DEFAULTS.dilate_iterations)
        self.merge_pad_spin.setValue(HOG_TUNING_DEFAULTS.merge_bbox_padding)
        self.lr_spin.setValue(HOG_TUNING_DEFAULTS.mog2_learning_rate)
        self.on_frames_spin.setValue(HOG_TUNING_DEFAULTS.motion_on_frames)
        self.off_frames_spin.setValue(HOG_TUNING_DEFAULTS.motion_off_frames)
        self.person_detector_combo.setCurrentIndex(0)
        self.on_detector_changed()
        self.single_box_cb.setChecked(True)
        self.cc_cb.setChecked(False)
        self.prob_smooth_cb.setChecked(False)
        self.prob_conf_cb.setChecked(False)
        self.debug_cb.setChecked(False)
        if hasattr(self, "roi_mode_combo"):
            self.roi_mode_combo.setCurrentIndex(0)
            self.on_roi_mode_changed()

    def on_detector_changed(self):
        detector_mode = (self.person_detector_combo.currentData() or "hog").strip().lower()
        show_hog = detector_mode == "hog"
        for label, widget in self._hog_setting_rows:
            label.setVisible(show_hog)
            widget.setVisible(show_hog)

    def open_video(self):
        project_root = Path(__file__).resolve().parents[2]
        default_data_dir = (project_root / "data" / "wisenet" / "video_sets").resolve()
        default_dir = str(default_data_dir if default_data_dir.is_dir() else Path.cwd())
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video",
            default_dir,
            "Video Files (*.mp4 *.avi)",
        )
        if path:
            self.state.input_path = Path(path)
            self.start_preview()
            self._update_run_enabled()

    def start_preview(self):
        if self.video_thread:
            self.video_thread.stop()
        self.video_thread = VideoThread(self.state.input_path)
        self.video_thread.change_pixmap_signal.connect(self.update_preview)
        self.video_thread.start()
        self.status_text.append(f"Opened video: {self.state.input_path}")

    def update_preview(self, qt_image: QImage):
        self.preview.set_image(QPixmap.fromImage(qt_image))
        if self.state.roi_mode == "auto":
            self._set_auto_roi_preview()

    def _compute_auto_roi_rect(self) -> tuple[int, int, int, int] | None:
        if self.preview.image.isNull():
            return None
        frame_width = int(self.preview.image.width())
        frame_height = int(self.preview.image.height())
        if frame_width <= 0 or frame_height <= 0:
            return None

        margin_x = int(round(frame_width * AUTO_ROI_MARGIN_RATIO))
        margin_y = int(round(frame_height * AUTO_ROI_MARGIN_RATIO))
        x1 = max(0, margin_x)
        y1 = max(0, margin_y)
        x2 = max(x1 + 1, frame_width - margin_x)
        y2 = max(y1 + 1, frame_height - margin_y)
        return (x1, y1, x2, y2)

    def _set_auto_roi_preview(self):
        auto_rect = self._compute_auto_roi_rect()
        if auto_rect is None:
            return
        self.preview.set_rois_frame([auto_rect])
        self.state.roi_rects = [auto_rect]

    def on_roi_selected(self, rois: list[tuple[int, int, int, int]]):
        self.state.roi_rects = list(rois)
        self.clear_roi_btn.setEnabled(bool(rois))
        self._update_run_enabled()
        self.status_text.append(f"ROI set: {len(rois)} rectangle(s)")

    def clear_rois(self):
        self.preview.clear_rois()
        self.state.roi_rects = None
        self.clear_roi_btn.setEnabled(False)
        self._update_run_enabled()
        self.status_text.append("ROI cleared")

    def _update_run_enabled(self):
        has_video = self.state.input_path is not None
        if self.state.roi_mode == "auto":
            self.run_btn.setEnabled(has_video)
            return
        has_roi = bool(self.state.roi_rects)
        self.run_btn.setEnabled(has_video and has_roi)

    def on_roi_mode_changed(self):
        roi_mode = str(self.roi_mode_combo.currentData() or "manual")
        self.state.roi_mode = roi_mode
        if roi_mode == "auto":
            self.preview.set_roi_enabled(False)
            self._set_auto_roi_preview()
            self.clear_roi_btn.setEnabled(False)
            self.status_text.append("ROI mode: automatic (full frame)")
        else:
            self.preview.set_roi_enabled(True)
            self.state.roi_rects = self.preview.get_rois()
            self.clear_roi_btn.setEnabled(bool(self.state.roi_rects))
            self.status_text.append("ROI mode: manual draw")
        self._update_run_enabled()

    def browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path:
            self.state.output_dir = Path(path)
            self.output_label.setText(str(self.state.output_dir))

    def run_detection(self):
        if not self.state.input_path:
            QMessageBox.warning(self, "Missing Info", "Please open a video first.")
            return

        if self.state.roi_mode == "auto":
            roi_rects = []
        else:
            if not self.state.roi_rects:
                QMessageBox.warning(self, "Missing Info", "Please set ROI first or switch to automatic ROI.")
                return
            roi_rects = list(self.state.roi_rects)

        detector_mode = self.person_detector_combo.currentData() or "hog"
        dnn_model_path = _find_default_dnn_model_path()
        if detector_mode == "dnn":
            if (
                dnn_model_path is None
                or not dnn_model_path.is_file()
                or not dnn_model_path.with_suffix(".prototxt").is_file()
                or not dnn_model_path.with_suffix(".caffemodel").is_file()
            ):
                QMessageBox.warning(
                    self,
                    "Missing DNN Model",
                    "Default MobileNet-SSD model not found in models/.",
                )
                return
        if detector_mode == "yolo":
            from ssc.features.yolo_person_detector import is_yolo_available

            if not is_yolo_available():
                QMessageBox.warning(
                    self,
                    "Missing YOLO Dependency",
                    "Ultralytics is not installed. Install requirements.txt and try again.",
                )
                return

        # Build configs
        self.state.motion_cfg = MotionConfig(
            min_contour_area=self.min_area_spin.value(),
            morph_close_iterations=self.close_iters_spin.value(),
            dilate_iterations=self.dilate_iters_spin.value(),
            merge_bbox_padding=self.merge_pad_spin.value(),
            mog2_learning_rate=self.lr_spin.value(),
            use_connected_components=self.cc_cb.isChecked(),
            single_box=self.single_box_cb.isChecked(),
            motion_on_frames=self.on_frames_spin.value(),
            motion_off_frames=self.off_frames_spin.value(),
            enable_smoothing=self.prob_smooth_cb.isChecked(),
            show_confidence=self.prob_conf_cb.isChecked(),
            roi_rects=tuple(roi_rects),
        )
        self.state.recorder_cfg = RecorderConfig(post_event_frames=self.post_frames_spin.value())
        self.state.person_detector = str(detector_mode)
        self.state.dnn_model_path = dnn_model_path

        # Stop preview
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None

        self.stop_event.clear()
        self.detection_thread = DetectionThread(
            input_path=self.state.input_path,
            output_dir=self.state.output_dir,
            motion_cfg=self.state.motion_cfg,
            recorder_cfg=self.state.recorder_cfg,
            stop_event=self.stop_event,
            debug=self.debug_cb.isChecked(),
            person_detector=self.state.person_detector,
            dnn_model_path=self.state.dnn_model_path,
        )
        self.detection_thread.finished.connect(self.on_detection_finished)
        self.detection_thread.status_update.connect(self.status_text.append)
        self.detection_thread.frame_annotated.connect(self.on_detection_frame)
        self.detection_thread.start()

        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress.setVisible(True)
        self.status_text.append("Detection started...")

    def stop_detection(self):
        self.stop_event.set()
        self.status_text.append("Cancelling...")

    def on_detection_finished(self, rc: int):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress.setVisible(False)
        if rc == 0:
            self.status_text.append("Detection completed successfully.")
        else:
            self.status_text.append("Detection stopped or failed.")
        self.alert_label.setText(ALERT_IDLE_TEXT)
        self.alert_label.setStyleSheet(ALERT_IDLE_STYLE)
        # Optionally restore original preview if desired
        if self.video_thread is None and self.state.input_path:
            self.start_preview()

    def on_detection_frame(self, qt_image: QImage, frame_idx: int, presence_p: float | None):
        # Show live detection feed in preview
        self.preview.set_image(QPixmap.fromImage(qt_image))

        if presence_p is not None and presence_p >= ALERT_PROB_THRESHOLD:
            self.alert_label.setText(ALERT_ACTIVE_TEXT)
            self.alert_label.setStyleSheet(ALERT_ACTIVE_STYLE)
        else:
            self.alert_label.setText(ALERT_IDLE_TEXT)
            self.alert_label.setStyleSheet(ALERT_IDLE_STYLE)
        # Optionally update a label with presence probability
        if presence_p is not None:
            self.status_text.append(f"Frame {frame_idx}: P(presence)={presence_p:.2f}")

    def open_outputs(self):
        import os
        import subprocess
        import sys

        out = self.state.output_dir.resolve()
        out.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(str(out))
        elif sys.platform == "darwin":
            subprocess.run(["open", str(out)])
        else:
            subprocess.run(["xdg-open", str(out)])

    def closeEvent(self, event):
        if self.video_thread:
            self.video_thread.stop()
        if self.detection_thread:
            self.stop_event.set()
            self.detection_thread.wait()
        event.accept()


def main() -> int:
    app = QApplication([])
    win = SmartSecurityCameraGUI()
    win.show()
    return app.exec()


if __name__ == "__main__":
    import sys
    sys.exit(main())
