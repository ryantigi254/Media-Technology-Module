from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ssc.config import RecorderConfig
from ssc.video_io import VideoInfo


@dataclass
class Incident:
    index: int
    start_frame: int
    last_motion_frame: int
    clip_path: Path
    meta_path: Path
    frames_written: int = 0


class IncidentRecorder:
    def __init__(self, output_dir: Path, video_info: VideoInfo, cfg: RecorderConfig):
        self.output_dir = output_dir
        self.video_info = video_info
        self.cfg = cfg

        self._writer: cv2.VideoWriter | None = None
        self._incident: Incident | None = None
        self._post_counter = 0
        self._incident_index = 0
        self._incident_durations_frames: list[int] = []
        self._first_incident_start_frame: int | None = None

        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def is_recording(self) -> bool:
        return self._writer is not None

    @property
    def incident_durations_frames(self) -> list[int]:
        return list(self._incident_durations_frames)

    @property
    def first_incident_start_frame(self) -> int | None:
        return self._first_incident_start_frame

    def _try_open_writer(self, clip_path: Path, fourcc: str) -> cv2.VideoWriter | None:
        writer = cv2.VideoWriter(
            str(clip_path),
            cv2.VideoWriter_fourcc(*fourcc),
            self.video_info.fps,
            (self.video_info.width, self.video_info.height),
        )
        if writer.isOpened():
            return writer
        writer.release()

        # Some OpenCV builds create an empty file even if the writer fails to open.
        try:
            if clip_path.exists():
                clip_path.unlink()
        except OSError:
            pass

        return None

    def _open_writer_with_fallbacks(self, clip_stem: str) -> tuple[cv2.VideoWriter, Path]:
        # codec support can be a bit inconsistent depending on OS / OpenCV build
        # (so this tries a couple of common options instead of just failing)
        candidates: list[tuple[str, str]] = [
            (".mp4", self.cfg.fourcc),
            (".avi", "XVID"),
            (".avi", "MJPG"),
        ]

        last_path = None
        for ext, fourcc in candidates:
            clip_path = self.output_dir / f"{clip_stem}{ext}"
            last_path = clip_path
            writer = self._try_open_writer(clip_path, fourcc)
            if writer is not None:
                return writer, clip_path

        raise RuntimeError(f"Could not open VideoWriter for {last_path} (check codec support)")

    def _start(self, frame_idx: int) -> None:
        self._incident_index += 1
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self._first_incident_start_frame is None:
            self._first_incident_start_frame = frame_idx

        clip_stem = f"incident_{self._incident_index:03d}_{stamp}"
        meta_path = self.output_dir / f"{clip_stem}.json"

        self._writer, clip_path = self._open_writer_with_fallbacks(clip_stem)

        self._incident = Incident(
            index=self._incident_index,
            start_frame=frame_idx,
            last_motion_frame=frame_idx,
            clip_path=clip_path,
            meta_path=meta_path,
        )

        # keep writing for a short time after motion stops
        # (otherwise you can miss the end of the "incident")
        self._post_counter = self.cfg.post_event_frames

    def _stop(self, end_frame: int) -> None:
        if self._writer is not None:
            self._writer.release()
        self._writer = None

        if self._incident is None:
            return

        self._incident_durations_frames.append(int(self._incident.frames_written))

        meta = {
            "incident_index": self._incident.index,
            "start_frame": self._incident.start_frame,
            "end_frame": end_frame,
            "frames_written": self._incident.frames_written,
            "video_fps": self.video_info.fps,
            "clip_path": str(self._incident.clip_path),
        }
        self._incident.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self._incident = None

    def update(self, frame_idx: int, frame_bgr_annotated: np.ndarray, motion: bool) -> None:
        if motion and not self.is_recording:
            # start a new incident clip on the first detected motion frame
            self._start(frame_idx)

        if self.is_recording:
            assert self._writer is not None
            assert self._incident is not None

            self._writer.write(frame_bgr_annotated)
            self._incident.frames_written += 1

            if motion:
                self._incident.last_motion_frame = frame_idx
                self._post_counter = self.cfg.post_event_frames
            else:
                self._post_counter -= 1
                if self._post_counter <= 0:
                    self._stop(end_frame=frame_idx)

    def close(self, final_frame_idx: int) -> None:
        if self.is_recording:
            self._stop(end_frame=final_frame_idx)
