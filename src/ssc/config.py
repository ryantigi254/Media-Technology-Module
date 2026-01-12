from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MotionConfig:
    min_contour_area: int = 800
    blur_ksize: int = 5
    mog2_history: int = 300
    mog2_var_threshold: float = 16.0
    mog2_detect_shadows: bool = True
    mog2_learning_rate: float = -1.0
    threshold_value: int = 200
    morph_kernel: int = 3
    morph_close_iterations: int = 2
    dilate_iterations: int = 2
    merge_bbox_padding: int = 12
    use_connected_components: bool = False
    single_box: bool = False
    motion_on_frames: int = 2
    motion_off_frames: int = 3
    enable_smoothing: bool = False
    show_confidence: bool = False
    roi_rects: tuple[tuple[int, int, int, int], ...] = ()


HOG_TUNING_DEFAULTS = MotionConfig(
    min_contour_area=1200,
    blur_ksize=5,
    mog2_history=300,
    mog2_var_threshold=16.0,
    mog2_detect_shadows=True,
    mog2_learning_rate=0.005,
    threshold_value=200,
    morph_kernel=3,
    morph_close_iterations=3,
    dilate_iterations=2,
    merge_bbox_padding=18,
    use_connected_components=False,
    single_box=False,
    motion_on_frames=3,
    motion_off_frames=6,
    enable_smoothing=False,
    show_confidence=False,
    roi_rects=(),
)


@dataclass(frozen=True)
class RecorderConfig:
    post_event_frames: int = 30
    fourcc: str = "mp4v"
