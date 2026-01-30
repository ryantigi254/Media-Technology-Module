# CSY3058 Smart Security Camera - Complete Project Code

Generated at: csy3058_as2_smart_security_camera
This document contains all source code and configuration files for the project.

## Table of Contents
- [.pytest_cache\README.md](#-pytest_cache-README-md)
- [docs\EVALUATION_REPORT 2 .md](#docs-EVALUATION_REPORT 2 -md)
- [environment.yml](#environment-yml)
- [gui.py](#gui-py)
- [qt_gui.py](#qt_gui-py)
- [README.MD](#README-MD)
- [requirements-dev.txt](#requirements-dev-txt)
- [requirements.txt](#requirements-txt)
- [requirements_frozen.txt](#requirements_frozen-txt)
- [run.py](#run-py)
- [src\ssc\__init__.py](#src-ssc-__init__-py)
- [src\ssc\__main__.py](#src-ssc-__main__-py)
- [src\ssc\cli.py](#src-ssc-cli-py)
- [src\ssc\config.py](#src-ssc-config-py)
- [src\ssc\gui_app.py](#src-ssc-gui_app-py)
- [src\ssc\misc\__init__.py](#src-ssc-misc-__init__-py)
- [src\ssc\misc\probabilistic_presence.py](#src-ssc-misc-probabilistic_presence-py)
- [src\ssc\motion.py](#src-ssc-motion-py)
- [src\ssc\pipeline.py](#src-ssc-pipeline-py)
- [src\ssc\qt_gui.py](#src-ssc-qt_gui-py)
- [src\ssc\recorder.py](#src-ssc-recorder-py)
- [src\ssc\video_io.py](#src-ssc-video_io-py)
- [tests\__init__.py](#tests-__init__-py)
- [tests\conftest.py](#tests-conftest-py)
- [tests\test_annotation.py](#tests-test_annotation-py)
- [tests\test_dnn_model_discovery.py](#tests-test_dnn_model_discovery-py)
- [tests\test_hog_integrity.py](#tests-test_hog_integrity-py)
- [tests\test_motion.py](#tests-test_motion-py)
- [tests\test_pipeline_e2e_synthetic.py](#tests-test_pipeline_e2e_synthetic-py)
- [tests\test_recorder.py](#tests-test_recorder-py)
- [tests\test_video_io.py](#tests-test_video_io-py)
- [tests\test_wisenet_eval.py](#tests-test_wisenet_eval-py)
- [tests\test_wisenet_integration.py](#tests-test_wisenet_integration-py)
- [tools\benchmark_runtime.py](#tools-benchmark_runtime-py)
- [tools\collect_project_code.py](#tools-collect_project_code-py)
- [tools\count_words_refined.py](#tools-count_words_refined-py)
- [tools\export_wisenet_predictions.py](#tools-export_wisenet_predictions-py)
- [tools\run_wisenet_dnn.py](#tools-run_wisenet_dnn-py)
- [tools\run_wisenet_hog.py](#tools-run_wisenet_hog-py)
- [tools\run_yolo.py](#tools-run_yolo-py)
- [tools\validate_hog_integrity.py](#tools-validate_hog_integrity-py)
- [tools\wisenet_class_balance.py](#tools-wisenet_class_balance-py)
- [tools\wisenet_eval.py](#tools-wisenet_eval-py)
- [tools\wisenet_split.py](#tools-wisenet_split-py)

---

## README.md <a id='-pytest_cache-README-md'></a>

```markdown
# pytest cache directory #

This directory contains data from the pytest's cache plugin,
which provides the `--lf` and `--ff` options, as well as the `cache` fixture.

**Do not** commit this to version control.

See [the docs](https://docs.pytest.org/en/stable/how-to/cache.html) for more information.

```

---

## EVALUATION_REPORT 2 .md <a id='docs-EVALUATION_REPORT 2 -md'></a>

```markdown

# Evaluation Report: WiseNET Person Detection

## 1. Experimental Setup

### Data Splits

* **Expanded Test Split** (`data/wisenet_split.json`): The primary test set consisting of 18 videos from sets 1, 2, 3, 4, 5, 9, and 11.
* **Option A Split** (`data/wisenet_split_option_a.json`): A balanced snapshot subset containing 8 videos from sets 4 and 9 only.

### Ground Truth (GT) Definitions

* **Manual Annotations (Human GT):** Human-labelled bounding boxes provided by the WiseNET dataset. This is the primary standard for accuracy.
* **Automatic Annotations (Auto):** Detector outputs (HOG_SVM, SSD_512, YOLOv3_608) provided with the dataset. Comparing against these measures agreement with those specific models, not necessarily ground truth correctness.

### Models Compared

1. **Our HOG Pipeline:** Standard Histogram of Oriented Gradients approach.
2. **Our DNN Pipeline:** MobileNet-SSD via OpenCV DNN.
3. **Dataset Reference Models:** Auto HOG_SVM, Auto SSD_512, Auto YOLOv3_608.

### Evaluation Method

* **Frame-level metrics:** Based on binary "person present" classification per frame.
* **Intersection over Union (IoU):** Computed on frames where both GT and prediction contain at least one box.
* **Entry/Exit Detection:** Compares the first/last GT-positive frames against the first/last predicted-positive frames.
* **Latency:** Negative values indicate early triggering (before GT); positive values indicate late triggering.

## 2. Configuration Parameters

The following **MotionConfig** defaults (`HOG_TUNING_DEFAULTS`) were used for the HOG test runs:

* `min_contour_area`: 1200
* `blur_ksize`: 5
* `mog2_history`: 300
* `mog2_var_threshold`: 16.0
* `mog2_detect_shadows`: true
* `mog2_learning_rate`: 0.005
* `threshold_value`: 200
* `morph_kernel`: 3
* `morph_close_iterations`: 3
* `dilate_iterations`: 2
* `merge_bbox_padding`: 18
* `use_connected_components`: false / `single_box`: false
* `motion_on_frames`: 3 / `motion_off_frames`: 6

## 3. Performance Results

### Master Comparison Table (Expanded Split)

*Note: `Prob=On` indicates probabilistic presence smoothing was enabled. `Prob=N/A` applies to dataset-provided annotations.*

| GT Source | Model             | Prob | Precision | Recall | F1     | Mean IoU         | IoU≥0.5         | Entry | Exit |
| :-------- | :---------------- | :--- | :-------- | :----- | :----- | :--------------- | :--------------- | :---- | :--- |
| Manual    | **Our DNN** | Off  | 0.9151    | 0.3274 | 0.4823 | **0.5065** | **0.6032** | 0.00  | 0.14 |
| Manual    | **Our HOG** | Off  | 0.9151    | 0.3274 | 0.4823 | 0.3560           | 0.1677           | 0.00  | 0.14 |
| Manual    | Our DNN           | On   | 0.9188    | 0.3171 | 0.4715 | 0.4041           | 0.4248           | 0.00  | 0.21 |
| Manual    | Our HOG           | On   | 0.9188    | 0.3171 | 0.4715 | 0.2571           | 0.1188           | 0.00  | 0.21 |
| Manual    | *Auto SSD_512*  | N/A  | 0.9970    | 0.7863 | 0.8792 | 0.7280           | 0.9565           | 0.14  | 0.36 |
| Manual    | *Auto YOLOv3*   | N/A  | 0.9967    | 0.7723 | 0.8703 | 0.7544           | 0.9741           | 0.21  | 0.50 |

### Agreement with Automatic Models (Expanded Split)

*Comparing our models directly against the dataset's pre-computed model outputs.*

| Comparison Pair                   | Precision | Recall | F1     | Mean IoU | IoU ≥ 0.5 |
| :-------------------------------- | :-------- | :----- | :----- | :------- | :--------- |
| **Our DNN vs Auto SSD_512** | 0.8226    | 0.3732 | 0.5135 | 0.5123   | 0.5868     |
| **Our DNN vs Auto YOLOv3**  | 0.8877    | 0.4099 | 0.5609 | 0.5287   | 0.6543     |
| **Our HOG vs Auto HOG_SVM** | 0.6278    | 0.5927 | 0.6097 | 0.2313   | 0.0150     |

## 4. Detailed Analysis

### Derived Metrics (Expanded Split)

*These metrics highlight the conservative nature of our pipeline (high Specificity/Precision, lower Recall).*

| Model             | Accuracy | Specificity | FPR    | FNR    | NPV    | Balanced Acc | MCC     | G-mean |
| :---------------- | :------- | :---------- | :----- | :----- | :----- | :----------- | :------ | :----- |
| **Our HOG** | 0.6599   | 0.9715      | 0.0285 | 0.6726 | 0.6065 | 0.6495       | 0.3949  | 0.5640 |
| **Our DNN** | 0.6599   | 0.9715      | 0.0285 | 0.6726 | 0.6065 | 0.6495       | 0.3949  | 0.5640 |
| *Auto SSD_512*  | 0.7845   | 0.0000      | 1.0000 | 0.2137 | 0.0000 | 0.3932       | -0.0253 | 0.0000 |

### Latency Analysis

*Measured in frames. Negative values indicate the model triggered before the Ground Truth (early trigger).*

| Model                 | Entry Latency (Mean) | Exit Latency (Mean) | Interpretation                 |
| :-------------------- | :------------------- | :------------------ | :----------------------------- |
| **Our HOG/DNN** | **-61.36**     | **-80.93**    | Early trigger / Early clearing |
| Auto SSD_512          | 52.14                | -68.43              | Late trigger / Early clearing  |
| Auto YOLOv3           | 48.71                | -62.00              | Late trigger / Early clearing  |

## 5. Key Observations & Interpretation

1. **Localisation Improvements:** While Our HOG and Our DNN share identical frame-level presence metrics (P/R/F1) due to shared motion-gating logic, the **DNN pipeline significantly outperforms HOG in localisation**.
   * DNN Mean IoU: ~0.51 (Expanded) vs HOG Mean IoU: ~0.36.
   * DNN IoU ≥ 0.5: ~60% vs HOG IoU ≥ 0.5: ~17%.
2. **Conservative Behaviour:** Our pipelines demonstrate high precision (>91%) and specificity (>97%) but modest recall (~33%). This indicates the system effectively suppresses false positives but may miss fainter or fleeting detections compared to the aggressive Auto SSD/YOLO models.
3. **Probabilistic Presence:** Enabling probabilistic smoothing slightly raises precision (0.9151 → 0.9188) but reduces recall and IoU. It forces a trade-off that may not be beneficial for this specific configuration.
4. **Entry/Exit Timing:** Our models exhibit negative latency (-61 frames entry), meaning they often trigger *before* the manual ground truth indicates a person is fully present. This suggests the motion detection component is highly sensitive to initial movement (e.g., shadows or door opening) before the person is fully visible.
5. **Comparison to State-of-the-Art:** The WiseNET automatic models (SSD/YOLO) achieve much higher recall (>70%) and IoU (>0.70). Our system functions as a lightweight, conservative detector suitable for different operational constraints.

## 6. Real-world Deployment Recommendations

* **Usage Strategy:** Treat this system as a **high-precision, conservative detector**. It is best used for verified escalation rather than primary safety-critical detection where missing a person is unacceptable.
* **Calibration:** Thresholds must be calibrated per site. The negative latency suggests the motion sensitivity might be too high for some environments, picking up precursors to events (shadows/doors) rather than the events themselves.
* **Human-in-the-loop:** Implement a review step for low-confidence events. Prioritize clips with consistent detections across multiple frames to leverage the system's high precision.
* **Drift Monitoring:** Re-evaluate performance monthly. Changes in lighting (seasonal) or camera angles will require re-tuning of the `mog2_var_threshold` and background history.

## 7. File References

* **Split Configs:** `data/wisenet_split.json`, `data/wisenet_split_option_a.json`
* **Evaluation Scripts:** `tools/wisenet_eval.py`, `tools/export_wisenet_predictions.py`
* **Metric Logs:** `evaluations/metrics/metrics_dnn_vs_manual.json`, `evaluations/metrics/metrics_hog_vs_manual.json`

```

---

## environment.yml <a id='environment-yml'></a>

```yaml
name: csy3058-smart-security-camera
dependencies:
  - python=3.10
  - pip
  - pip:
  - bleach==6.3.0
  - certifi==2026.1.4
  - charset-normalizer==3.4.4
  - colorama==0.4.6
  - contourpy==1.3.2
  - cycler==0.12.1
  - exceptiongroup==1.3.1
  - filelock==3.20.3
  - fonttools==4.61.1
  - fsspec==2026.1.0
  - idna==3.11
  - iniconfig==2.3.0
  - Jinja2==3.1.6
  - kaggle==1.7.4.5
  - kiwisolver==1.4.9
  - MarkupSafe==3.0.3
  - matplotlib==3.10.8
  - mpmath==1.3.0
  - networkx==3.4.2
  - numpy==2.2.6
  - opencv-python==4.13.0.90
  - packaging==25.0
  - pillow==12.1.0
  - pluggy==1.6.0
  - polars==1.37.1
  - polars-runtime-32==1.37.1
  - protobuf==6.33.4
  - psutil==7.2.1
  - Pygments==2.19.2
  - pyparsing==3.3.2
  - PySide6==6.10.1
  - PySide6_Addons==6.10.1
  - PySide6_Essentials==6.10.1
  - pytest==9.0.2
  - python-dateutil==2.9.0.post0
  - python-slugify==8.0.4
  - PyYAML==6.0.3
  - requests==2.32.5
  - scipy==1.15.3
  - shiboken6==6.10.1
  - six==1.17.0
  - sympy==1.14.0
  - text-unidecode==1.3
  - tomli==2.4.0
  - torch==2.9.1
  - torchvision==0.24.1
  - tqdm==4.67.1
  - typing_extensions==4.15.0
  - ultralytics==8.4.6
  - ultralytics-thop==2.0.18
  - urllib3==2.6.3
  - webencodings==0.5.1

```

---

## gui.py <a id='gui-py'></a>

```python
from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_src_path() -> None:
    root = Path(__file__).resolve().parent
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main() -> int:
    _bootstrap_src_path()

    from ssc.gui_app import main as gui_main

    return gui_main()


if __name__ == "__main__":
    raise SystemExit(main())

```

---

## qt_gui.py <a id='qt_gui-py'></a>

```python
from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_src_path() -> None:
    root = Path(__file__).resolve().parent
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main() -> int:
    _bootstrap_src_path()
    from ssc.qt_gui import main as qt_main
    return qt_main()


if __name__ == "__main__":
    raise SystemExit(main())

```

---

## README.MD <a id='README-MD'></a>

```markdown
# CSY3058 Smart Security Camera System

This project implements a Python-based smart security camera pipeline with motion gating, person verification (HOG/DNN), and evidence recording.

## 🛑 Perfect Environment Setup (Required)

To run this system without errors, you **MUST** use the exact dependency versions listed below.

### 1. Create Environment

**Option A: Conda (Recommended)**
```bash
conda create -n csy3058-camera python=3.10
conda activate csy3058-camera
```

**Option B: Venv**
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install Exact Dependencies
Create a file named `requirements_frozen.txt` with the following content, then run `pip install -r requirements_frozen.txt`.

**requirements_frozen.txt content:**
```
bleach==6.3.0
certifi==2026.1.4
charset-normalizer==3.4.4
colorama==0.4.6
contourpy==1.3.2
cycler==0.12.1
exceptiongroup==1.3.1
filelock==3.20.3
fonttools==4.61.1
fsspec==2026.1.0
idna==3.11
iniconfig==2.3.0
Jinja2==3.1.6
kaggle==1.7.4.5
kiwisolver==1.4.9
MarkupSafe==3.0.3
matplotlib==3.10.8
mpmath==1.3.0
networkx==3.4.2
numpy==2.2.6
opencv-python==4.13.0.90
packaging==25.0
pillow==12.1.0
pluggy==1.6.0
polars==1.37.1
polars-runtime-32==1.37.1
protobuf==6.33.4
psutil==7.2.1
Pygments==2.19.2
pyparsing==3.3.2
PySide6==6.10.1
PySide6_Addons==6.10.1
PySide6_Essentials==6.10.1
pytest==9.0.2
python-dateutil==2.9.0.post0
python-slugify==8.0.4
PyYAML==6.0.3
requests==2.32.5
scipy==1.15.3
shiboken6==6.10.1
six==1.17.0
sympy==1.14.0
text-unidecode==1.3
tomli==2.4.0
torch==2.9.1
torchvision==0.24.1
tqdm==4.67.1
typing_extensions==4.15.0
ultralytics==8.4.6
ultralytics-thop==2.0.18
urllib3==2.6.3
webencodings==0.5.1
```

Run installation:
```bash
pip install -r requirements_frozen.txt
```

---

## 🚀 How to Run the System

### 1. Graphical User Interface (GUI) - **Main Run Mode**
This is the primary way to use the system. It allows you to load videos, tune thresholds, and see real-time detection results.

```bash
python qt_gui.py
```
*Note: This opens a window where you can load a video and adjust sliders.*

### 2. Standard Run (Command Line / Automated)
Use this command to process a video file without the GUI, useful for batch processing.

```bash
# Basic run with display
python run.py --input "data/raw/your_video.mp4" --output "outputs" --display

# With specific min-area (sensitivity)
python run.py --input "data/raw/your_video.mp4" --output "outputs" --min-area 800 --display
```

### 3. YOLO Demonstration (Verification Only)
To test the optional YOLO backend:
```bash
python tools/run_yolo.py --input "data/wisenet/video_sets/set_4/video4_4.avi" --display
```

---

## 🧪 How to Test

Run the full test suite to verify the installation and pipeline logic:

```bash
# Run unit tests
pytest

# Run integration tests (requires WiseNET data in 'data/wisenet')
pytest -m integration
```

If your tests pass, the system is correctly configured.

```

---

## requirements-dev.txt <a id='requirements-dev-txt'></a>

```text
pytest>=7.4
scikit-image

```

---

## requirements.txt <a id='requirements-txt'></a>

```text
opencv-python
numpy
PySide6
ultralytics

```

---

## requirements_frozen.txt <a id='requirements_frozen-txt'></a>

```text
bleach==6.3.0
certifi==2026.1.4
charset-normalizer==3.4.4
colorama==0.4.6
contourpy==1.3.2
cycler==0.12.1
exceptiongroup==1.3.1
filelock==3.20.3
fonttools==4.61.1
fsspec==2026.1.0
idna==3.11
iniconfig==2.3.0
Jinja2==3.1.6
kaggle==1.7.4.5
kiwisolver==1.4.9
MarkupSafe==3.0.3
matplotlib==3.10.8
mpmath==1.3.0
networkx==3.4.2
numpy==2.2.6
opencv-python==4.13.0.90
packaging==25.0
pillow==12.1.0
pluggy==1.6.0
polars==1.37.1
polars-runtime-32==1.37.1
protobuf==6.33.4
psutil==7.2.1
Pygments==2.19.2
pyparsing==3.3.2
PySide6==6.10.1
PySide6_Addons==6.10.1
PySide6_Essentials==6.10.1
pytest==9.0.2
python-dateutil==2.9.0.post0
python-slugify==8.0.4
PyYAML==6.0.3
requests==2.32.5
scipy==1.15.3
shiboken6==6.10.1
six==1.17.0
sympy==1.14.0
text-unidecode==1.3
tomli==2.4.0
torch==2.9.1
torchvision==0.24.1
tqdm==4.67.1
typing_extensions==4.15.0
ultralytics==8.4.6
ultralytics-thop==2.0.18
urllib3==2.6.3
webencodings==0.5.1

```

---

## run.py <a id='run-py'></a>

```python
from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_src_path() -> None:
    root = Path(__file__).resolve().parent
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main() -> int:
    _bootstrap_src_path()

    from ssc.cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())

```

---

## __init__.py <a id='src-ssc-__init__-py'></a>

```python
"""Smart Security Camera (CSY3058 AS2)."""

__all__ = ["__version__"]

__version__ = "0.1.0"

```

---

## __main__.py <a id='src-ssc-__main__-py'></a>

```python
from __future__ import annotations

from ssc.cli import main


if __name__ == "__main__":
    raise SystemExit(main())

```

---

## cli.py <a id='src-ssc-cli-py'></a>

```python
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

```

---

## config.py <a id='src-ssc-config-py'></a>

```python
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

```

---

## gui_app.py <a id='src-ssc-gui_app-py'></a>

```python
from __future__ import annotations

import os
import threading
from pathlib import Path
from tkinter import BooleanVar, DoubleVar, IntVar, StringVar, Tk
from tkinter import filedialog, messagebox, ttk

import cv2

from ssc.config import MotionConfig, RecorderConfig, HOG_TUNING_DEFAULTS
from ssc.features.dnn_person_detector import find_default_dnn_model_path
from ssc.pipeline import process_video


def _default_dnn_model_path() -> str:
    project_root = Path(__file__).resolve().parents[2]
    default_path = find_default_dnn_model_path(project_root)
    return str(default_path) if default_path is not None else ""


class App:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("CSY3058 Smart Security Camera")

        # keeping the GUI state in Tk variables so widgets update nicely

        self.input_path = StringVar()
        self.output_dir = StringVar(value=str(Path("outputs").resolve()))
        self.min_area = IntVar(value=HOG_TUNING_DEFAULTS.min_contour_area)
        self.post_frames = IntVar(value=30)
        self.close_iters = IntVar(value=HOG_TUNING_DEFAULTS.morph_close_iterations)
        self.dilate_iters = IntVar(value=HOG_TUNING_DEFAULTS.dilate_iterations)
        self.merge_pad = IntVar(value=HOG_TUNING_DEFAULTS.merge_bbox_padding)
        self.learning_rate = DoubleVar(value=HOG_TUNING_DEFAULTS.mog2_learning_rate)
        self.on_frames = IntVar(value=HOG_TUNING_DEFAULTS.motion_on_frames)
        self.off_frames = IntVar(value=HOG_TUNING_DEFAULTS.motion_off_frames)
        self.roi_rects = StringVar(value="")
        self.prob_draw = BooleanVar(value=True)
        self.single_box = BooleanVar(value=True)
        self.use_cc = BooleanVar(value=True)
        self.display = BooleanVar(value=False)
        self.show_mask = BooleanVar(value=False)
        self.person_detector = StringVar(value="motion")

        self.status = StringVar(value="idle")

        self._widgets_to_disable: list[ttk.Widget] = []
        self._progress: ttk.Progressbar | None = None
        self._stop_event = threading.Event()
        self.stop_btn: ttk.Button | None = None

        self._build()

    def _build(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        container = ttk.Frame(self.root, padding=12)
        container.grid(row=0, column=0, sticky="nsew")

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        input_box = ttk.LabelFrame(container, text="Input", padding=10)
        input_box.grid(row=0, column=0, sticky="ew")
        input_box.columnconfigure(1, weight=1)

        ttk.Label(input_box, text="Video file").grid(row=0, column=0, sticky="w")
        input_entry = ttk.Entry(input_box, textvariable=self.input_path)
        input_entry.grid(row=0, column=1, sticky="ew", padx=(8, 8))
        input_browse = ttk.Button(input_box, text="Browse", command=self._browse_input)
        input_browse.grid(row=0, column=2)

        ttk.Label(input_box, text="Output folder").grid(row=1, column=0, sticky="w", pady=(8, 0))
        out_entry = ttk.Entry(input_box, textvariable=self.output_dir)
        out_entry.grid(row=1, column=1, sticky="ew", padx=(8, 8), pady=(8, 0))
        out_browse = ttk.Button(input_box, text="Browse", command=self._browse_output)
        out_browse.grid(row=1, column=2, pady=(8, 0))

        settings_box = ttk.LabelFrame(container, text="Settings", padding=10)
        settings_box.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        settings_box.columnconfigure(1, weight=1)

        ttk.Label(settings_box, text="min-area").grid(row=0, column=0, sticky="w")
        min_area_spin = ttk.Spinbox(settings_box, from_=50, to=50000, textvariable=self.min_area, width=12)
        min_area_spin.grid(row=0, column=1, sticky="w", padx=(8, 0))

        ttk.Label(settings_box, text="post-frames").grid(row=1, column=0, sticky="w", pady=(8, 0))
        post_spin = ttk.Spinbox(settings_box, from_=0, to=500, textvariable=self.post_frames, width=12)
        post_spin.grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(8, 0))

        ttk.Label(settings_box, text="close-iters").grid(row=2, column=0, sticky="w", pady=(8, 0))
        close_spin = ttk.Spinbox(settings_box, from_=0, to=10, textvariable=self.close_iters, width=12)
        close_spin.grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(8, 0))

        ttk.Label(settings_box, text="dilate-iters").grid(row=3, column=0, sticky="w", pady=(8, 0))
        dilate_spin = ttk.Spinbox(settings_box, from_=0, to=10, textvariable=self.dilate_iters, width=12)
        dilate_spin.grid(row=3, column=1, sticky="w", padx=(8, 0), pady=(8, 0))

        ttk.Label(settings_box, text="merge-pad").grid(row=4, column=0, sticky="w", pady=(8, 0))
        merge_spin = ttk.Spinbox(settings_box, from_=0, to=100, textvariable=self.merge_pad, width=12)
        merge_spin.grid(row=4, column=1, sticky="w", padx=(8, 0), pady=(8, 0))

        ttk.Label(settings_box, text="learning-rate").grid(row=5, column=0, sticky="w", pady=(8, 0))
        lr_spin = ttk.Spinbox(settings_box, from_=-1.0, to=1.0, increment=0.01, textvariable=self.learning_rate, width=12)
        lr_spin.grid(row=5, column=1, sticky="w", padx=(8, 0), pady=(8, 0))

        ttk.Label(settings_box, text="on-frames").grid(row=6, column=0, sticky="w", pady=(8, 0))
        on_spin = ttk.Spinbox(settings_box, from_=1, to=30, textvariable=self.on_frames, width=12)
        on_spin.grid(row=6, column=1, sticky="w", padx=(8, 0), pady=(8, 0))

        ttk.Label(settings_box, text="off-frames").grid(row=7, column=0, sticky="w", pady=(8, 0))
        off_spin = ttk.Spinbox(settings_box, from_=1, to=60, textvariable=self.off_frames, width=12)
        off_spin.grid(row=7, column=1, sticky="w", padx=(8, 0), pady=(8, 0))

        ttk.Label(settings_box, text="roi (x1,y1,x2,y2;...)").grid(row=8, column=0, sticky="w", pady=(8, 0))
        roi_entry = ttk.Entry(settings_box, textvariable=self.roi_rects)
        roi_entry.grid(row=8, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))

        roi_btn = ttk.Button(settings_box, text="Select ROI...", command=self._pick_roi)
        roi_btn.grid(row=8, column=2, sticky="w", padx=(8, 0), pady=(8, 0))

        ttk.Label(settings_box, text="person-detector").grid(row=9, column=0, sticky="w", pady=(8, 0))
        detector_combo = ttk.Combobox(
            settings_box,
            textvariable=self.person_detector,
            state="readonly",
            values=["motion", "hog", "dnn"],
            width=12,
        )
        detector_combo.grid(row=9, column=1, sticky="w", padx=(8, 0), pady=(8, 0))

        single_cb = ttk.Checkbutton(settings_box, text="Single box mode", variable=self.single_box)
        single_cb.grid(row=10, column=0, columnspan=2, sticky="w", pady=(10, 0))

        cc_cb = ttk.Checkbutton(settings_box, text="Use connected components", variable=self.use_cc)
        cc_cb.grid(row=11, column=0, columnspan=2, sticky="w")

        prob_cb = ttk.Checkbutton(settings_box, text="Probabilistic smoothing (draw only)", variable=self.prob_draw)
        prob_cb.grid(row=12, column=0, columnspan=2, sticky="w")

        display_cb = ttk.Checkbutton(settings_box, text="Display preview (OpenCV)", variable=self.display)
        display_cb.grid(row=13, column=0, columnspan=2, sticky="w")
        mask_cb = ttk.Checkbutton(settings_box, text="Show motion mask", variable=self.show_mask)
        mask_cb.grid(row=14, column=0, columnspan=2, sticky="w")

        actions_box = ttk.Frame(container)
        actions_box.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        actions_box.columnconfigure(0, weight=1)

        self.run_btn = ttk.Button(actions_box, text="Run detection", command=self._run)
        self.run_btn.grid(row=0, column=0, sticky="ew")

        self.stop_btn = ttk.Button(actions_box, text="Stop / Cancel", command=self._cancel, state="disabled")
        self.stop_btn.grid(row=1, column=0, sticky="ew", pady=(6, 0))

        open_btn = ttk.Button(actions_box, text="Open outputs folder", command=self._open_output_folder)
        open_btn.grid(row=2, column=0, sticky="ew", pady=(6, 0))

        status_box = ttk.LabelFrame(container, text="Status", padding=10)
        status_box.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        status_box.columnconfigure(0, weight=1)

        self._progress = ttk.Progressbar(status_box, mode="indeterminate")
        self._progress.grid(row=0, column=0, sticky="ew")

        ttk.Label(status_box, textvariable=self.status).grid(row=1, column=0, sticky="w", pady=(8, 0))

        self._widgets_to_disable = [
            input_entry,
            input_browse,
            out_entry,
            out_browse,
            min_area_spin,
            post_spin,
            close_spin,
            dilate_spin,
            merge_spin,
            lr_spin,
            on_spin,
            off_spin,
            roi_entry,
            roi_btn,
            detector_combo,
            single_cb,
            cc_cb,
            prob_cb,
            display_cb,
            mask_cb,
            open_btn,
        ]

    def _pick_roi(self) -> None:
        inp = Path(self.input_path.get()).expanduser()
        if not inp.exists():
            messagebox.showerror("Missing input", "Please select a valid video file first.")
            return

        cap = cv2.VideoCapture(str(inp))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            messagebox.showerror("ROI", "Could not read first frame from the selected video.")
            return

        rois: list[tuple[int, int, int, int]] = []
        while True:
            x, y, w, h = cv2.selectROI("Select ROI (ENTER=accept, ESC=cancel)", frame, showCrosshair=True)
            if w == 0 or h == 0:
                break
            rois.append((int(x), int(y), int(x + w), int(y + h)))

            keep = messagebox.askyesno("ROI", "Add another ROI?")
            if not keep:
                break

        cv2.destroyWindow("Select ROI (ENTER=accept, ESC=cancel)")

        if rois:
            self.roi_rects.set("; ".join(f"{x1},{y1},{x2},{y2}" for x1, y1, x2, y2 in rois))

    def _browse_input(self) -> None:
        initial = None
        try:
            default = Path(__file__).resolve().parents[3] / "data"
            if default.exists():
                initial = str(default)
        except Exception:
            initial = None

        path = filedialog.askopenfilename(
            title="Select a video file",
            initialdir=initial,
            filetypes=[("Video files", "*.avi *.mp4"), ("All files", "*.*")],
        )
        if path:
            self.input_path.set(path)

    def _browse_output(self) -> None:
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.output_dir.set(path)


    def _open_output_folder(self) -> None:
        try:
            out = Path(self.output_dir.get()).expanduser().resolve()
            out.mkdir(parents=True, exist_ok=True)

            if os.name == "nt":
                os.startfile(str(out))
            else:
                messagebox.showinfo("Output folder", str(out))
        except Exception as e:
            messagebox.showerror("Could not open folder", str(e))

    def _set_running(self, running: bool) -> None:
        self.run_btn.configure(state=("disabled" if running else "normal"))
        if self.stop_btn is not None:
            self.stop_btn.configure(state=("normal" if running else "disabled"))
        for w in self._widgets_to_disable:
            w.configure(state=("disabled" if running else "normal"))

        if self._progress is not None:
            if running:
                self._progress.start(10)
            else:
                self._progress.stop()

    def _run(self) -> None:
        inp = Path(self.input_path.get()).expanduser()
        out = Path(self.output_dir.get()).expanduser()

        if not inp.exists():
            messagebox.showerror("Missing input", "Please select a valid video file.")
            return

        out.mkdir(parents=True, exist_ok=True)

        roi_rects: list[tuple[int, int, int, int]] = []
        raw = self.roi_rects.get().strip()
        if raw:
            for item in raw.split(";"):
                s = item.strip()
                if not s:
                    continue
                parts = [p.strip() for p in s.split(",")]
                if len(parts) != 4:
                    messagebox.showerror("Invalid ROI", "ROI must be x1,y1,x2,y2;... (integers)")
                    return
                try:
                    x1, y1, x2, y2 = (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]))
                except ValueError:
                    messagebox.showerror("Invalid ROI", "ROI must be x1,y1,x2,y2;... (integers)")
                    return
                roi_rects.append((x1, y1, x2, y2))

        motion_cfg = MotionConfig(
            min_contour_area=int(self.min_area.get()),
            morph_close_iterations=int(self.close_iters.get()),
            dilate_iterations=int(self.dilate_iters.get()),
            merge_bbox_padding=int(self.merge_pad.get()),
            mog2_learning_rate=float(self.learning_rate.get()),
            use_connected_components=bool(self.use_cc.get()),
            single_box=bool(self.single_box.get()),
            motion_on_frames=int(self.on_frames.get()),
            motion_off_frames=int(self.off_frames.get()),
            prob_presence_draw=bool(self.prob_draw.get()),
            roi_rects=tuple(roi_rects),
        )
        recorder_cfg = RecorderConfig(post_event_frames=int(self.post_frames.get()))

        detector_mode = (self.person_detector.get() or "motion").strip().lower()
        if detector_mode not in ("motion", "hog", "dnn"):
            detector_mode = "motion"

        dnn_path_text = _default_dnn_model_path()
        dnn_model_path = Path(dnn_path_text).expanduser() if dnn_path_text else None
        if detector_mode == "dnn":
            if (
                dnn_model_path is None
                or not dnn_model_path.is_file()
                or not dnn_model_path.with_suffix(".prototxt").is_file()
                or not dnn_model_path.with_suffix(".caffemodel").is_file()
            ):
                messagebox.showerror(
                    "Missing DNN Model",
                    "Default MobileNet-SSD model not found in models/.",
                )
                return

        self.status.set("running...")
        self._stop_event.clear()
        self._set_running(True)

        def worker() -> None:
            # run in a thread so the UI doesn't freeze while processing frames
            try:
                process_video(
                    input_path=inp,
                    output_dir=out,
                    motion_cfg=motion_cfg,
                    recorder_cfg=recorder_cfg,
                    display=bool(self.display.get()),
                    show_mask=bool(self.show_mask.get()),
                    stop_event=self._stop_event,
                    person_detector=detector_mode,
                    dnn_model_path=dnn_model_path,
                )
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Run failed", str(e)))
                self.root.after(0, lambda: self.status.set("failed"))
            else:
                if self._stop_event.is_set():
                    self.root.after(0, lambda: self.status.set("cancelled"))
                else:
                    self.root.after(0, lambda: self.status.set("done (check outputs folder)"))
            finally:
                self.root.after(0, lambda: self._set_running(False))

        threading.Thread(target=worker, daemon=True).start()

    def _cancel(self) -> None:
        if not self._stop_event.is_set():
            self._stop_event.set()
            self.status.set("cancelling...")


def main() -> int:
    root = Tk()
    _ = App(root)
    root.mainloop()
    return 0

```

---

## __init__.py <a id='src-ssc-misc-__init__-py'></a>

```python

```

---

## probabilistic_presence.py <a id='src-ssc-misc-probabilistic_presence-py'></a>

```python
from __future__ import annotations

from ssc.features.probabilistic_presence import (
    ProbabilisticPresenceTracker,
    ProbPresenceConfig,
    ProbPresenceState,
)

__all__ = [
    "ProbabilisticPresenceTracker",
    "ProbPresenceConfig",
    "ProbPresenceState",
]

```

---

## motion.py <a id='src-ssc-motion-py'></a>

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from ssc.config import MotionConfig

# References:
# - MOG2 background subtraction: https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html
# - Morphological operations: https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html


@dataclass(frozen=True)
class MotionResult:
    motion: bool
    bboxes: list[tuple[int, int, int, int]]
    motion_mask: np.ndarray


def _merge_bboxes(
    bboxes: list[tuple[int, int, int, int]],
    *,
    padding: int,
) -> list[tuple[int, int, int, int]]:
    if not bboxes:
        return []

    # Expand boxes slightly so nearby blobs (e.g., legs/arms) get merged.
    boxes = []
    for x, y, w, h in bboxes:
        x1 = x - padding
        y1 = y - padding
        x2 = x + w + padding
        y2 = y + h + padding
        boxes.append([x1, y1, x2, y2])

    merged: list[list[int]] = []
    used = [False] * len(boxes)

    for i in range(len(boxes)):
        if used[i]:
            continue

        x1, y1, x2, y2 = boxes[i]
        used[i] = True

        changed = True
        while changed:
            changed = False
            for j in range(len(boxes)):
                if used[j]:
                    continue

                a1, b1, a2, b2 = boxes[j]
                overlap = not (x2 < a1 or a2 < x1 or y2 < b1 or b2 < y1)
                if overlap:
                    x1 = min(x1, a1)
                    y1 = min(y1, b1)
                    x2 = max(x2, a2)
                    y2 = max(y2, b2)
                    used[j] = True
                    changed = True

        merged.append([x1, y1, x2, y2])

    out: list[tuple[int, int, int, int]] = []
    for x1, y1, x2, y2 in merged:
        out.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))

    return out


def _union_bbox(bboxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    if not bboxes:
        return []

    x1 = min(x for x, _, _, _ in bboxes)
    y1 = min(y for _, y, _, _ in bboxes)
    x2 = max(x + w for x, _, w, _ in bboxes)
    y2 = max(y + h for _, y, _, h in bboxes)

    return [(int(x1), int(y1), int(x2 - x1), int(y2 - y1))]


def _largest_bbox(bboxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    if not bboxes:
        return []
    x, y, w, h = max(bboxes, key=lambda b: int(b[2]) * int(b[3]))
    return [(int(x), int(y), int(w), int(h))]


def _find_bboxes_contours(mask: np.ndarray, min_area: int) -> list[tuple[int, int, int, int]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes: list[tuple[int, int, int, int]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # quick filter to ignore tiny flicker/noise (shadows, compression artifacts, etc.)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        bboxes.append((int(x), int(y), int(w), int(h)))

    return bboxes


def _find_bboxes_connected_components(mask: np.ndarray, min_area: int) -> list[tuple[int, int, int, int]]:
    # Connected components can be more stable on some clips, but may be stricter depending
    # on how the mask breaks up.
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    bboxes: list[tuple[int, int, int, int]] = []
    for label_idx in range(1, num_labels):
        x, y, w, h, area = stats[label_idx]
        if int(area) < min_area:
            continue
        bboxes.append((int(x), int(y), int(w), int(h)))

    return bboxes


class MotionDetector:
    def __init__(self, cfg: MotionConfig):
        self.cfg = cfg
        # using MOG2 since it works well for fixed CCTV-style cameras
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=cfg.mog2_history,
            varThreshold=cfg.mog2_var_threshold,
            detectShadows=cfg.mog2_detect_shadows,
        )

        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cfg.morph_kernel, cfg.morph_kernel)
        )

        self._roi_mask: np.ndarray | None = None
        self._roi_shape: tuple[int, int] | None = None

    def _get_roi_mask(self, shape_hw: tuple[int, int]) -> np.ndarray | None:
        if not self.cfg.roi_rects:
            return None

        if self._roi_mask is not None and self._roi_shape == shape_hw:
            return self._roi_mask

        h, w = shape_hw
        mask = np.zeros((h, w), dtype=np.uint8)
        for x1, y1, x2, y2 in self.cfg.roi_rects:
            ix1 = max(int(x1), 0)
            iy1 = max(int(y1), 0)
            ix2 = min(int(x2), w)
            iy2 = min(int(y2), h)
            if ix2 > ix1 and iy2 > iy1:
                mask[iy1:iy2, ix1:ix2] = 255

        self._roi_mask = mask
        self._roi_shape = shape_hw
        return self._roi_mask

    def detect(self, frame_bgr: np.ndarray) -> MotionResult:
        assert frame_bgr is not None and getattr(frame_bgr, "size", 0) > 0
        # background subtraction works best on a static camera
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.cfg.blur_ksize > 1:
            # blur smooths out small pixel noise so the mask is less "speckled"
            gray = cv2.GaussianBlur(gray, (self.cfg.blur_ksize, self.cfg.blur_ksize), 0)

        # learningRate controls how fast the background model updates.
        # Lower values can help stop a moving person being "absorbed" into the background.
        fg = self._subtractor.apply(gray, learningRate=float(self.cfg.mog2_learning_rate))

        # threshold helps get a clean binary mask for contour detection
        _, thresh = cv2.threshold(fg, self.cfg.threshold_value, 255, cv2.THRESH_BINARY)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self._kernel, iterations=1)
        if self.cfg.morph_close_iterations > 0:
            thresh = cv2.morphologyEx(
                thresh, cv2.MORPH_CLOSE, self._kernel, iterations=self.cfg.morph_close_iterations
            )
        # dilation makes blobs a bit more connected (boxes are less jittery)
        if self.cfg.dilate_iterations > 0:
            thresh = cv2.dilate(thresh, self._kernel, iterations=self.cfg.dilate_iterations)

        roi_mask = self._get_roi_mask(thresh.shape[:2])
        if roi_mask is not None:
            thresh = cv2.bitwise_and(thresh, roi_mask)

        if self.cfg.use_connected_components:
            bboxes = _find_bboxes_connected_components(thresh, min_area=self.cfg.min_contour_area)
        else:
            bboxes = _find_bboxes_contours(thresh, min_area=self.cfg.min_contour_area)
        bboxes = _merge_bboxes(bboxes, padding=self.cfg.merge_bbox_padding)
        if self.cfg.single_box:
            bboxes = _largest_bbox(bboxes)

        return MotionResult(motion=len(bboxes) > 0, bboxes=bboxes, motion_mask=thresh)

```

---

## pipeline.py <a id='src-ssc-pipeline-py'></a>

```python
from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

import cv2

from ssc.config import MotionConfig, RecorderConfig
from ssc.features.probabilistic_presence import ProbabilisticPresenceTracker
from ssc.features.dnn_person_detector import detect_people_dnn
from ssc.features.yolo_person_detector import detect_objects_yolo
from ssc.motion import MotionDetector
from ssc.recorder import IncidentRecorder
from ssc.video_io import get_video_info, open_capture


def _annotate(
    frame_bgr,
    frame_idx: int,
    bboxes: list[tuple[int, int, int, int]],
    motion: bool,
    presence_p: float | None = None,
):
    out = frame_bgr.copy()

    frame_h, frame_w = out.shape[:2]

    for (x, y, bw, bh) in bboxes:
        # bbox merging adds padding so coordinates can go outside the frame a bit
        x1 = max(int(x), 0)
        y1 = max(int(y), 0)
        x2 = min(int(x + bw), frame_w - 1)
        y2 = min(int(y + bh), frame_h - 1)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

    label = "MOTION" if motion else "idle"
    cv2.putText(
        out,
        f"{label} | frame={frame_idx}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0) if motion else (200, 200, 200),
        2,
        cv2.LINE_AA,
    )

    if presence_p is not None:
        cv2.putText(
            out,
            f"P(presence)={presence_p:.2f}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if motion else (200, 200, 200),
            2,
            cv2.LINE_AA,
        )

    return out


PERSON_INTERVAL_HOG = 3
PERSON_INTERVAL_DNN = 5
PERSON_INTERVAL_YOLO = 5

# The motion/YOLO gating alignment was AI-assisted to mirror GUI and CLI behaviour.


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


def _filter_to_roi(
    bboxes: list[tuple[int, int, int, int]],
    roi_rects: tuple[tuple[int, int, int, int], ...],
) -> list[tuple[int, int, int, int]]:
    if not roi_rects:
        return list(bboxes)
    filtered: list[tuple[int, int, int, int]] = []
    for x, y, w, h in bboxes:
        cx = float(x) + float(w) * 0.5
        cy = float(y) + float(h) * 0.5
        for rx1, ry1, rx2, ry2 in roi_rects:
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


def _safe_name(s: str) -> str:
    keep = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep)
    return out.strip("_") or "unknown"


def _guess_set_name(input_path: Path) -> str:
    parts = [p.lower() for p in input_path.parts]
    try:
        i = parts.index("data")
        if i + 1 < len(input_path.parts):
            return _safe_name(input_path.parts[i + 1])
    except ValueError:
        pass

    parent = input_path.parent.name
    return _safe_name(parent)


def _make_run_dir(*, base_output_dir: Path, input_path: Path) -> Path:
    set_name = _guess_set_name(input_path)
    video_name = _safe_name(input_path.stem)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_output_dir / set_name / video_name / f"run_{stamp}"


def process_video(
    *,
    input_path: Path,
    output_dir: Path,
    motion_cfg: MotionConfig,
    recorder_cfg: RecorderConfig,
    display: bool,
    show_mask: bool,
    stop_event: threading.Event | None = None,
    max_frames: int | None = None,
    person_detector: str = "motion",
    dnn_model_path: Path | None = None,
) -> int:
    cap = open_capture(input_path)
    info = get_video_info(cap)

    run_dir = _make_run_dir(base_output_dir=output_dir, input_path=input_path)
    run_dir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "input_path": str(input_path),
        "output_dir": str(run_dir),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "video_info": {
            "fps": info.fps,
            "width": info.width,
            "height": info.height,
            "frame_count": info.frame_count,
        },
        "motion_cfg": motion_cfg.__dict__,
        "recorder_cfg": recorder_cfg.__dict__,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    # detector finds motion blobs; recorder decides when to start/stop incident clips
    detector = MotionDetector(motion_cfg)
    recorder = IncidentRecorder(output_dir=run_dir, video_info=info, cfg=recorder_cfg)

    prob_tracker = ProbabilisticPresenceTracker() if (motion_cfg.enable_smoothing or motion_cfg.show_confidence) else None

    detector_mode = (person_detector or "motion").strip().lower()
    if detector_mode not in ("hog", "dnn", "yolo", "motion"):
        detector_mode = "motion"

    if detector_mode == "dnn":
        if (
            dnn_model_path is None
            or not dnn_model_path.is_file()
            or not dnn_model_path.with_suffix(".prototxt").is_file()
            or not dnn_model_path.with_suffix(".caffemodel").is_file()
        ):
            raise ValueError("DNN mode requires a valid .prototxt/.caffemodel model path.")

    hog = None
    if detector_mode == "hog":
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    if detector_mode == "hog":
        person_interval = PERSON_INTERVAL_HOG
    elif detector_mode == "dnn":
        person_interval = PERSON_INTERVAL_DNN
    elif detector_mode == "yolo":
        person_interval = PERSON_INTERVAL_YOLO
    else:
        person_interval = PERSON_INTERVAL_DNN
    last_person_frame_idx = -10_000
    last_person_bboxes: list[tuple[int, int, int, int]] = []

    frame_idx = -1
    processed = 0
    motion_frames = 0
    stable_motion = False
    on_counter = 0
    off_counter = 0
    cancelled = False
    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                cancelled = True
                break

            ok, frame = cap.read()
            if not ok:
                break

            frame_idx += 1
            res = detector.detect(frame)

            on_frames = max(int(motion_cfg.motion_on_frames), 1)
            off_frames = max(int(motion_cfg.motion_off_frames), 1)

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

            effective_motion = stable_motion and raw_motion

            if detector_mode == "yolo":
                if (frame_idx - last_person_frame_idx) >= person_interval:
                    last_person_bboxes = detect_objects_yolo(frame)
                    last_person_bboxes = _filter_to_roi(last_person_bboxes, motion_cfg.roi_rects)
                    last_person_frame_idx = frame_idx
            elif stable_motion and raw_motion and detector_mode in ("hog", "dnn"):
                if (frame_idx - last_person_frame_idx) >= person_interval:
                    if detector_mode == "hog":
                        last_person_bboxes = []
                        if hog is not None:
                            rois = list(motion_cfg.roi_rects)
                            if not rois:
                                rois = [(0, 0, int(info.width), int(info.height))]
                            for x1, y1, x2, y2 in rois:
                                ix1 = max(int(x1), 0)
                                iy1 = max(int(y1), 0)
                                ix2 = min(int(x2), int(info.width))
                                iy2 = min(int(y2), int(info.height))
                                if ix2 <= ix1 or iy2 <= iy1:
                                    continue

                                roi = frame[iy1:iy2, ix1:ix2]
                                if roi.size == 0:
                                    continue

                                rh, rw = roi.shape[:2]
                                if rw < 64 or rh < 128:
                                    continue

                                scale = 1.0
                                target_w = 640
                                if rw > target_w:
                                    scale = float(target_w) / float(rw)
                                    roi = cv2.resize(roi, (int(round(rw * scale)), int(round(rh * scale))))

                                rects, _ = hog.detectMultiScale(
                                    roi,
                                    winStride=(8, 8),
                                    padding=(8, 8),
                                    scale=1.05,
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
                                        last_person_bboxes.append(
                                            (int(fx1), int(fy1), int(fx2 - fx1), int(fy2 - fy1))
                                        )
                    else:
                        assert dnn_model_path is not None
                        last_person_bboxes = detect_people_dnn(frame, dnn_model_path)
                        last_person_bboxes = _filter_to_roi(last_person_bboxes, motion_cfg.roi_rects)
                    last_person_frame_idx = frame_idx
            elif detector_mode != "yolo":
                last_person_bboxes = []

            # --- Selection Logic ---
            p_val: float | None = None
            display_bboxes: list[tuple[int, int, int, int]] = []
            
            # 1. Update Tracker (if active for either smoothing or confidence)
            if prob_tracker is not None:
                detected_bbox = None
                # Tracker input selection
                if detector_mode == "yolo" and last_person_bboxes:
                    candidate = last_person_bboxes
                elif detector_mode in ("hog", "dnn") and last_person_bboxes and raw_motion:
                    candidate = _filter_to_roi(last_person_bboxes, motion_cfg.roi_rects)
                elif effective_motion:
                    candidate = _filter_to_roi(res.bboxes, motion_cfg.roi_rects)
                else:
                    candidate = []

                if candidate:
                    detected_bbox = candidate[0] if len(candidate) == 1 else _union_xywh(candidate)

                _ = prob_tracker.update(detected_bbox=detected_bbox)
                if motion_cfg.show_confidence:
                    p_val = prob_tracker.probability()

            # 2. Decide Display Boxes
            if motion_cfg.enable_smoothing and prob_tracker is not None:
                # Use Smoothed Boxes
                draw_motion = prob_tracker.should_draw()
                draw_bbox = prob_tracker.bbox()
                display_bboxes = [draw_bbox] if (draw_motion and draw_bbox is not None) else []
            else:
                # Use Raw Boxes
                if detector_mode == "yolo" and last_person_bboxes:
                    display_bboxes = last_person_bboxes
                elif stable_motion and raw_motion:
                    if detector_mode in ("hog", "dnn") and last_person_bboxes:
                        display_bboxes = _filter_to_roi(last_person_bboxes, motion_cfg.roi_rects)
                    else:
                        display_bboxes = _filter_to_roi(res.bboxes, motion_cfg.roi_rects)
                else:
                    display_bboxes = []
            
            # 3. Annotate
            should_draw_motion = bool(display_bboxes) or (stable_motion and raw_motion)
            annotated = _annotate(frame, frame_idx, display_bboxes, should_draw_motion, presence_p=p_val)

            recorder.update(frame_idx, annotated, motion=effective_motion)

            if effective_motion:
                motion_frames += 1

            if display:
                cv2.imshow("smart-security-camera", annotated)
                if show_mask:
                    cv2.imshow("motion-mask", res.motion_mask)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            processed += 1
            # max_frames is mainly for unit/integration tests so they don't run for ages
            if max_frames is not None and processed >= max_frames:
                break

    finally:
        recorder.close(final_frame_idx=frame_idx)
        cap.release()
        if display:
            cv2.destroyAllWindows()

        durations_frames = recorder.incident_durations_frames
        incident_count = len(durations_frames)
        durations_sec = [float(frames) / float(info.fps) for frames in durations_frames if info.fps > 0]
        avg_duration_sec = float(sum(durations_sec) / len(durations_sec)) if durations_sec else 0.0
        max_duration_sec = float(max(durations_sec)) if durations_sec else 0.0

        startup_window_frames = int(round(float(info.fps))) if info.fps > 0 else 0
        first_start = recorder.first_incident_start_frame
        false_start_incident = False
        if (
            first_start is not None
            and startup_window_frames > 0
            and first_start < startup_window_frames
            and durations_frames
            and int(durations_frames[0]) < startup_window_frames
        ):
            false_start_incident = True

        run_summary = {
            "processed_frames": int(processed),
            "motion_frames": int(motion_frames),
            "motion_frames_ratio": (float(motion_frames) / float(processed)) if processed > 0 else 0.0,
            "incident_count": int(incident_count),
            "avg_incident_duration_sec": float(avg_duration_sec),
            "max_incident_duration_sec": float(max_duration_sec),
            "startup_window_frames": int(startup_window_frames),
            "false_start_incident": bool(false_start_incident),
            "cancelled": bool(cancelled),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        }

        (run_dir / "run_summary.json").write_text(
            json.dumps(run_summary, indent=2), encoding="utf-8"
        )

        run_meta["run_summary"] = run_summary
        (run_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    return 1 if cancelled else 0

```

---

## qt_gui.py <a id='src-ssc-qt_gui-py'></a>

```python
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

```

---

## recorder.py <a id='src-ssc-recorder-py'></a>

```python
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

```

---

## video_io.py <a id='src-ssc-video_io-py'></a>

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    width: int
    height: int
    frame_count: int


def open_capture(video_path: Path) -> cv2.VideoCapture:
    # OpenCV will return a closed capture if the path/codec is wrong
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    return cap


def get_video_info(cap: cv2.VideoCapture) -> VideoInfo:
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    if fps <= 1e-6:
        # some AVI files don't report FPS properly, so default to something sensible
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    return VideoInfo(fps=fps, width=width, height=height, frame_count=frame_count)

```

---

## __init__.py <a id='tests-__init__-py'></a>

```python

# These test files were AI-assisted to help track performance during iterative development.
# Reason: tests were used to guide implementation quality during the assignment build.

```

---

## conftest.py <a id='tests-conftest-py'></a>

```python
from __future__ import annotations

import sys
from pathlib import Path

import pytest


root = Path(__file__).resolve().parents[1]
src = root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))


def write_synthetic_video(
    path: Path,
    *,
    width: int = 160,
    height: int = 120,
    fps: int = 10,
    total_frames: int = 40,
    move_start: int = 10,
) -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    path.parent.mkdir(parents=True, exist_ok=True)

    # AVI + MJPG tends to be the most portable option in OpenCV on Windows.
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter for synthetic test video")

    try:
        for i in range(total_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)

            # simple moving block so motion is guaranteed
            if i >= move_start:
                x = 10 + (i - move_start) * 3
                x = int(min(max(x, 0), width - 30))
                cv2.rectangle(frame, (x, 40), (x + 20, 70), (255, 255, 255), -1)

            writer.write(frame)
    finally:
        writer.release()

```

---

## test_annotation.py <a id='tests-test_annotation-py'></a>

```python
from __future__ import annotations

import pytest

_ = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from ssc.pipeline import _annotate


def test_annotate_draws_bbox() -> None:
    frame = np.zeros((80, 120, 3), dtype=np.uint8)

    # bbox is (x, y, w, h)
    annotated = _annotate(frame, frame_idx=0, bboxes=[(20, 30, 10, 10)], motion=True)

    # rectangle is green in BGR
    assert annotated[30, 20].tolist() == [0, 255, 0]

```

---

## test_dnn_model_discovery.py <a id='tests-test_dnn_model_discovery-py'></a>

```python
from __future__ import annotations

from pathlib import Path

from ssc.features.dnn_person_detector import find_default_dnn_model_path


def test_find_default_dnn_model_path_returns_proto(tmp_path: Path) -> None:
    proto = tmp_path / "model.prototxt"
    model = tmp_path / "model.caffemodel"
    proto.write_text("proto", encoding="utf-8")
    model.write_text("model", encoding="utf-8")

    found = find_default_dnn_model_path(tmp_path)

    assert found == proto


def test_find_default_dnn_model_path_none(tmp_path: Path) -> None:
    found = find_default_dnn_model_path(tmp_path)

    assert found is None

```

---

## test_hog_integrity.py <a id='tests-test_hog_integrity-py'></a>

```python
from __future__ import annotations

import cv2
import numpy as np


def test_hog_descriptor_dimension_matches_compute() -> None:
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    expected = int(hog.getDescriptorSize())

    # Standard pedestrian window size for HOG people detector.
    win_w, win_h = 64, 128
    frame = np.zeros((win_h, win_w), dtype=np.uint8)
    feats = hog.compute(frame)
    assert feats is not None
    assert int(feats.shape[0]) == expected

```

---

## test_motion.py <a id='tests-test_motion-py'></a>

```python
from __future__ import annotations

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from ssc.config import MotionConfig
from ssc.motion import MotionDetector


def test_motion_detector_no_motion_after_learning() -> None:
    cfg = MotionConfig(
        min_contour_area=50,
        blur_ksize=1,
        mog2_history=20,
        mog2_detect_shadows=False,
        threshold_value=127,
        morph_close_iterations=0,
        dilate_iterations=0,
        single_box=False,
    )
    det = MotionDetector(cfg)

    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    # let the subtractor settle
    for _ in range(25):
        _ = det.detect(frame)

    res = det.detect(frame)
    assert res.motion is False


def test_motion_detector_detects_moving_object() -> None:
    cfg = MotionConfig(
        min_contour_area=50,
        blur_ksize=1,
        mog2_history=20,
        mog2_detect_shadows=False,
        threshold_value=127,
        morph_close_iterations=0,
        dilate_iterations=0,
        single_box=False,
    )
    det = MotionDetector(cfg)

    base = np.zeros((120, 160, 3), dtype=np.uint8)

    for _ in range(20):
        _ = det.detect(base)

    frame = base.copy()
    cv2.rectangle(frame, (40, 40), (70, 80), (255, 255, 255), -1)

    res = det.detect(frame)
    assert res.motion is True
    assert len(res.bboxes) >= 1

```

---

## test_pipeline_e2e_synthetic.py <a id='tests-test_pipeline_e2e_synthetic-py'></a>

```python
from __future__ import annotations

from pathlib import Path

import pytest

_ = pytest.importorskip("cv2")

from ssc.config import MotionConfig, RecorderConfig
from ssc.pipeline import process_video


def test_pipeline_end_to_end_on_synthetic_video(tmp_path: Path) -> None:
    from tests.conftest import write_synthetic_video

    video = tmp_path / "synthetic.avi"
    write_synthetic_video(video, total_frames=40, move_start=10)

    out_dir = tmp_path / "outputs"

    motion_cfg = MotionConfig(
        min_contour_area=50,
        blur_ksize=1,
        mog2_history=20,
        mog2_detect_shadows=False,
        threshold_value=127,
    )
    recorder_cfg = RecorderConfig(post_event_frames=5)

    code = process_video(
        input_path=video,
        output_dir=out_dir,
        motion_cfg=motion_cfg,
        recorder_cfg=recorder_cfg,
        display=False,
        show_mask=False,
        max_frames=40,
    )
    assert code == 0

    clips = list(out_dir.rglob("*.mp4")) + list(out_dir.rglob("*.avi"))
    metas = list(out_dir.rglob("*.json"))

    assert len(clips) >= 1
    assert len(metas) >= 1

```

---

## test_recorder.py <a id='tests-test_recorder-py'></a>

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

_ = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from ssc.config import RecorderConfig
from ssc.recorder import IncidentRecorder
from ssc.video_io import VideoInfo


def test_incident_recorder_writes_clip_and_metadata(tmp_path: Path) -> None:
    info = VideoInfo(fps=10.0, width=160, height=120, frame_count=0)
    rec = IncidentRecorder(output_dir=tmp_path, video_info=info, cfg=RecorderConfig(post_event_frames=5))

    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    # trigger recording
    for i in range(3):
        rec.update(i, frame, motion=True)

    # allow it to stop
    for j in range(3, 12):
        rec.update(j, frame, motion=False)

    rec.close(final_frame_idx=12)

    clips = sorted([p for p in tmp_path.iterdir() if p.suffix.lower() in {".mp4", ".avi"}])
    metas = sorted([p for p in tmp_path.iterdir() if p.suffix.lower() == ".json"])

    assert len(clips) == 1
    assert len(metas) == 1

    meta = json.loads(metas[0].read_text(encoding="utf-8"))
    assert meta["frames_written"] > 0
    assert Path(meta["clip_path"]).exists()

```

---

## test_video_io.py <a id='tests-test_video_io-py'></a>

```python
from __future__ import annotations

from pathlib import Path

import pytest

_ = pytest.importorskip("cv2")

from ssc.video_io import get_video_info, open_capture


def test_open_capture_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "nope.avi"
    with pytest.raises(FileNotFoundError):
        _ = open_capture(missing)


def test_open_capture_and_get_info(tmp_path: Path) -> None:
    from tests.conftest import write_synthetic_video

    vid = tmp_path / "synthetic.avi"
    write_synthetic_video(vid, width=160, height=120, fps=10, total_frames=15)

    cap = open_capture(vid)
    try:
        info = get_video_info(cap)
        assert info.width == 160
        assert info.height == 120
        assert info.fps > 0
    finally:
        cap.release()

```

---

## test_wisenet_eval.py <a id='tests-test_wisenet_eval-py'></a>

```python
from __future__ import annotations

import json
from pathlib import Path

from tools.wisenet_eval import evaluate_video, load_detections, FrameDetections


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_evaluate_video_basic_metrics(tmp_path: Path) -> None:
    gt_path = tmp_path / "gt.json"
    pred_path = tmp_path / "pred.json"

    gt_payload = {
        "frames": [
            {"frame_index": 0, "bboxes": [[0, 0, 10, 10]]},
            {"frame_index": 1, "bboxes": [[0, 0, 10, 10]]},
            {"frame_index": 2, "bboxes": []},
        ]
    }
    pred_payload = {
        "frames": [
            {"frame_index": 0, "bboxes": [[0, 0, 10, 10]]},
            {"frame_index": 1, "bboxes": []},
            {"frame_index": 2, "bboxes": [[0, 0, 10, 10]]},
        ]
    }

    _write_json(gt_path, gt_payload)
    _write_json(pred_path, pred_payload)

    gt = load_detections(gt_path)
    pred = load_detections(pred_path)

    metrics = evaluate_video(
        video_name="video1",
        gt=gt,
        pred=pred,
        fps=30.0,
        entry_tolerance=0,
        exit_tolerance=0,
    )

    assert metrics.tp == 1
    assert metrics.fn == 1
    assert metrics.fp == 1
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.f1 == 0.5
    assert metrics.mean_iou == 1.0
    assert metrics.iou_50_rate == 1.0


def test_load_detections_accepts_detection_list(tmp_path: Path) -> None:
    gt_path = tmp_path / "gt.json"
    payload = [
        {"frame": 3, "detections": [{"x": 1, "y": 2, "w": 3, "h": 4}]},
        {"frame": 4, "detections": [{"xmin": 0, "ymin": 0, "xmax": 5, "ymax": 5}]},
    ]
    _write_json(gt_path, payload)

    detections = load_detections(gt_path)

    assert detections.by_frame[3] == [(1.0, 2.0, 3.0, 4.0)]
    assert detections.by_frame[4] == [(0.0, 0.0, 5.0, 5.0)]

```

---

## test_wisenet_integration.py <a id='tests-test_wisenet_integration-py'></a>

```python
from __future__ import annotations

import os
from pathlib import Path

import pytest

_ = pytest.importorskip("cv2")

from ssc.config import MotionConfig, RecorderConfig
from ssc.pipeline import process_video


@pytest.mark.integration
def test_wisenet_smoke(tmp_path: Path) -> None:
    # This is a smoke test: it checks we can open and process a real WiseNET AVI.
    # It intentionally doesn't assert that motion MUST be found in the first N frames.

    env_dir = os.environ.get("WISENET_DIR")
    if env_dir:
        root = Path(env_dir)
    else:
        # default matches your local download: <repo_root>/data
        repo_root = Path(__file__).resolve().parents[2]
        root = repo_root / "data"

    if not root.exists():
        pytest.skip(f"WiseNET dataset folder not found: {root}")

    videos = sorted(root.rglob("*.avi"))
    if not videos:
        pytest.skip(f"No .avi files found under: {root}")

    video = videos[0]

    out_dir = tmp_path / "wisenet_outputs"

    code = process_video(
        input_path=video,
        output_dir=out_dir,
        motion_cfg=MotionConfig(min_contour_area=800),
        recorder_cfg=RecorderConfig(post_event_frames=10),
        display=False,
        show_mask=False,
        max_frames=60,
    )

    assert code == 0
    assert out_dir.exists()

```

---

## benchmark_runtime.py <a id='tools-benchmark_runtime-py'></a>

```python
"""Runtime benchmark for the smart security camera pipeline.

This script was generated with assistance and is intended to provide a repeatable,
transparent way to measure throughput (FPS) for the different detection backends.

It benchmarks the *analytics* path (decode + motion gate + optional person model),
and intentionally avoids recorder/overlay I/O so the reported FPS reflects the
core processing cost.
"""

from __future__ import annotations

# This file was AI assisted as it is mostly for evaluations and testing purposes.

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

```

---

## collect_project_code.py <a id='tools-collect_project_code-py'></a>

```python
from __future__ import annotations
import os
from pathlib import Path

# Configuration: Files to include
INCLUDE_EXTS = {".py", ".md", ".txt", ".yml", ".json"}
EXCLUDE_DIRS = {
    ".git", "__pycache__", "outputs", "data", "models", 
    ".idea", ".vscode", ".venv", "envs", "features", "wandb",
    "evaluations" # Skip raw json evals, keep summary? Maybe skip large jsons.
}
EXCLUDE_FILES = {
    "package-lock.json", "yarn.lock", "Poetry.lock", 
    "wisenet_split.json", "Project_Code_Complete.md"
}

def is_text_file(path: Path) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            f.read(1024)
        return True
    except UnicodeDecodeError:
        return False

def collect_files(root: Path) -> list[Path]:
    files = []
    for root_dir, dirs, filenames in os.walk(root):
        # Filter directories inplace
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        
        for name in filenames:
            path = Path(root_dir) / name
            if path.suffix.lower() not in INCLUDE_EXTS:
                continue
            if name in EXCLUDE_FILES:
                continue
            # Skip large files (e.g. > 1MB)
            if path.stat().st_size > 1_000_000:
                print(f"Skipping large file: {path}")
                continue
            
            files.append(path)
    return sorted(files)

def main():
    root = Path(".")
    output_file = root / "Project_Code_Complete.md"
    
    files = collect_files(root)
    
    with open(output_file, "w", encoding="utf-8") as out:
        out.write("# CSY3058 Smart Security Camera - Complete Project Code\n\n")
        out.write(f"Generated at: {os.path.basename(os.getcwd())}\n")
        out.write("This document contains all source code and configuration files for the project.\n\n")
        
        out.write("## Table of Contents\n")
        for f in files:
            anchor = str(f).replace('\\', '-').replace('/', '-').replace('.', '-')
            out.write(f"- [{f}](#{anchor})\n")
        out.write("\n---\n\n")
        
        for f in files:
            try:
                content = f.read_text(encoding="utf-8")
                # Determine language for markdown syntax
                ext = f.suffix.lower()
                lang = "text"
                if ext == ".py": lang = "python"
                if ext == ".md": lang = "markdown"
                if ext == ".json": lang = "json"
                if ext == ".yml": lang = "yaml"
                # Safe anchor generation
                anchor = str(f).replace('\\', '-').replace('/', '-').replace('.', '-')
                
                out.write(f"## {f.name} <a id='{anchor}'></a>\n\n")
                out.write(f"```{lang}\n")
                out.write(content)
                out.write("\n```\n\n")
                out.write("---\n\n")
            except Exception as e:
                print(f"Error reading {f}: {e}")

    print(f"Successfully aggregated {len(files)} files into {output_file}")

if __name__ == "__main__":
    main()

```

---

## count_words_refined.py <a id='tools-count_words_refined-py'></a>

```python
import re
from pathlib import Path

def count_words_refined(file_path):
    text = Path(file_path).read_text(encoding='utf-8')
    
    text = re.sub(r'%.*', '', text)
    match = re.search(r'\\begin\{document\}(.*?)\\end\{document\}', text, re.DOTALL)
    body = match.group(1) if match else text

    # Exclude Code Blocks
    body = re.sub(r'\\begin\{lstlisting\}.*?\\end\{lstlisting\}', ' ', body, flags=re.DOTALL)

    # Exclude Titles/TOC
    body = re.sub(r'\\maketitle', '', body)
    body = re.sub(r'\\tableofcontents', '', body)
    body = re.sub(r'\\listoffigures', '', body)
    body = re.sub(r'\\listoftables', '', body)
    body = re.sub(r'\\newpage', '', body)
    body = re.sub(r'\\clearpage', '', body)
    body = re.sub(r'\\begin\{abstract\}', '', body)
    body = re.sub(r'\\end\{abstract\}', '', body)

    # Exclude Captions/Labels/Images
    def remove_command_content(text, command_name):
        result = []
        idx = 0
        cmd_len = len(command_name)
        while idx < len(text):
            if text[idx:idx+cmd_len] == command_name:
                peek_idx = idx + cmd_len
                while peek_idx < len(text) and text[peek_idx].isspace(): peek_idx += 1
                if peek_idx < len(text) and text[peek_idx] == '{':
                    brace_count = 1
                    curr = peek_idx + 1
                    while curr < len(text) and brace_count > 0:
                        if text[curr] == '{': brace_count += 1
                        elif text[curr] == '}': brace_count -= 1
                        curr += 1
                    idx = curr
                    continue
            result.append(text[idx])
            idx += 1
        return "".join(result)

    body = remove_command_content(body, "\\caption")
    body = remove_command_content(body, "\\label")
    body = remove_command_content(body, "\\includegraphics")
    
    body = re.sub(r'\\bibliography\{.*?\}', '', body)
    body = re.sub(r'\\bibliographystyle\{.*?\}', '', body)
    
    body = re.sub(r'\$.*?\$', ' ', body)
    body = re.sub(r'\\begin\{equation\}.*?\\end\{equation\}', ' ', body, flags=re.DOTALL)
    
    body = re.sub(r'\\[a-zA-Z]+\*?', ' ', body)
    body = re.sub(r'[\{\}\[\]]', ' ', body)
    
    words = [w for w in body.split() if any(c.isalnum() for c in w)]
    return len(words)

if __name__ == "__main__":
    p = r"e:\22837352\Media-Technology-Module\csy3058_as2_smart_security_camera\docs\report\CSY3058_A2_Report.tex"
    c = count_words_refined(p)
    Path("word_count_result.txt").write_text(str(c), encoding="utf-8")

```

---

## export_wisenet_predictions.py <a id='tools-export_wisenet_predictions-py'></a>

```python
from __future__ import annotations

# This file was AI assisted as it is mostly for evaluations and testing purposes.

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from ssc.config import HOG_TUNING_DEFAULTS, MotionConfig
from ssc.features.dnn_person_detector import find_default_dnn_model_path, detect_people_dnn
from ssc.features.probabilistic_presence import ProbabilisticPresenceTracker
from ssc.motion import MotionDetector
from ssc.video_io import open_capture, get_video_info

# These evaluation/export scripts were AI-assisted for scaffolding and then verified.
# Reason: they evaluate the implementation but are not required by the brief.


@dataclass(frozen=True)
class FrameRecord:
    frame_number: int
    detections: list[dict[str, Any]]
    presence_probability: float | None = None


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


def _hog_people(frame_bgr: np.ndarray, width: int, height: int, hog: cv2.HOGDescriptor) -> list[tuple[int, int, int, int]]:
    # Reference: Dalal & Triggs HOG (CVPR 2005) https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf
    out: list[tuple[int, int, int, int]] = []
    roi = frame_bgr
    rh, rw = roi.shape[:2]
    if rw < 64 or rh < 128:
        return out

    scale = 1.0
    target_w = 640
    if rw > target_w:
        scale = float(target_w) / float(rw)
        roi = cv2.resize(roi, (int(round(rw * scale)), int(round(rh * scale))))

    rects, _ = hog.detectMultiScale(
        roi,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05,
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

        fx1 = max(fx - px, 0)
        fy1 = max(fy - py, 0)
        fx2 = min(fx + fw + px, int(width))
        fy2 = min(fy + fh + py, int(height))

        if fx2 > fx1 and fy2 > fy1:
            out.append((int(fx1), int(fy1), int(fx2 - fx1), int(fy2 - fy1)))

    return out


def _frame_to_records(
    frame_idx: int,
    bboxes: list[tuple[int, int, int, int]],
    algo: str,
    presence_probability: float | None,
) -> FrameRecord:
    # We record per-frame presence probability to compare against GT presence segments.
    # This is optional and only present when probabilistic tracking is enabled.
    detections = [
        {
            "class": "person",
            "imageAlgorithm": algo,
            "xywh": [int(x), int(y), int(w), int(h)],
        }
        for (x, y, w, h) in bboxes
    ]
    return FrameRecord(
        frame_number=frame_idx,
        detections=detections,
        presence_probability=presence_probability,
    )


def _export_video(
    *,
    video_path: Path,
    output_path: Path,
    motion_cfg: MotionConfig,
    detector_mode: str,
    dnn_model_path: Path | None,
    max_frames: int | None,
    prob: bool,
) -> None:
    cap = open_capture(video_path)
    info = get_video_info(cap)

    detector = MotionDetector(motion_cfg)
    prob_tracker = ProbabilisticPresenceTracker() if prob else None

    hog = None
    if detector_mode == "hog":
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    person_interval = 3 if detector_mode == "hog" else 5
    last_person_frame_idx = -10_000
    last_person_bboxes: list[tuple[int, int, int, int]] = []

    stable_motion = False
    on_counter = 0
    off_counter = 0

    frames: list[dict[str, Any]] = []
    frame_idx = -1
    processed = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_idx += 1
            res = detector.detect(frame)

            on_frames = max(int(motion_cfg.motion_on_frames), 1)
            off_frames = max(int(motion_cfg.motion_off_frames), 1)

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

            if stable_motion and raw_motion and detector_mode in ("hog", "dnn"):
                if (frame_idx - last_person_frame_idx) >= person_interval:
                    if detector_mode == "hog" and hog is not None:
                        last_person_bboxes = _hog_people(frame, info.width, info.height, hog)
                    elif detector_mode == "dnn" and dnn_model_path is not None:
                        last_person_bboxes = detect_people_dnn(frame, dnn_model_path)
                    else:
                        last_person_bboxes = []
                    last_person_frame_idx = frame_idx
            else:
                last_person_bboxes = []

            if prob_tracker is not None:
                detected_bbox = None
                if detector_mode in ("hog", "dnn") and last_person_bboxes and raw_motion:
                    candidate = last_person_bboxes
                elif stable_motion and raw_motion:
                    candidate = res.bboxes
                else:
                    candidate = []

                if candidate:
                    detected_bbox = candidate[0] if len(candidate) == 1 else _union_xywh(candidate)

                _ = prob_tracker.update(detected_bbox=detected_bbox)
                draw_motion = prob_tracker.should_draw()
                draw_bbox = prob_tracker.bbox()
                bboxes = [draw_bbox] if (draw_motion and draw_bbox is not None) else []
                presence_probability = prob_tracker.probability() if prob_tracker is not None else None
                record = _frame_to_records(frame_idx, bboxes, detector_mode, presence_probability)
            else:
                if stable_motion and raw_motion:
                    if detector_mode in ("hog", "dnn") and last_person_bboxes:
                        bboxes = last_person_bboxes
                    else:
                        bboxes = res.bboxes
                else:
                    bboxes = []
                record = _frame_to_records(frame_idx, bboxes, detector_mode, None)

            frame_payload: dict[str, Any] = {
                "frameNumber": record.frame_number,
                "detections": record.detections,
            }
            if record.presence_probability is not None:
                frame_payload["presence_probability"] = float(record.presence_probability)
            frames.append(frame_payload)

            processed += 1
            if max_frames is not None and processed >= max_frames:
                break
    finally:
        cap.release()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"frames": frames}, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export per-frame predictions for WiseNET videos.")
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("docs") / "project" / "wisenet_split.json",
        help="Split JSON path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluations") / "predictions",
        help="Base output directory for prediction JSON files.",
    )
    parser.add_argument(
        "--detector",
        choices=["hog", "dnn", "motion"],
        default="hog",
        help="Detector to export (hog/dnn/motion).",
    )
    parser.add_argument("--max-videos", type=int, default=None, help="Limit number of videos.")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames per video.")
    parser.add_argument("--prob", action="store_true", help="Enable probabilistic smoothing.")

    args = parser.parse_args()
    test_items = _load_test_items(args.split)
    videos = list(_iter_videos(test_items))
    if args.max_videos is not None:
        videos = videos[: max(0, int(args.max_videos))]

    dnn_model_path = None
    if args.detector == "dnn":
        dnn_model_path = find_default_dnn_model_path(Path(__file__).resolve().parents[1])
        if dnn_model_path is None:
            raise SystemExit("Default MobileNet-SSD model not found in models/.")

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

    for video_path in videos:
        if not video_path.exists():
            continue
        out_path = args.output / args.detector / f"{video_path.stem}.json"
        _export_video(
            video_path=video_path,
            output_path=out_path,
            motion_cfg=motion_cfg,
            detector_mode=args.detector,
            dnn_model_path=dnn_model_path,
            max_frames=args.max_frames,
            prob=bool(args.prob),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

```

---

## run_wisenet_dnn.py <a id='tools-run_wisenet_dnn-py'></a>

```python
from __future__ import annotations

# This file was AI assisted as it is mostly for evaluations and testing purposes.

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from ssc.config import HOG_TUNING_DEFAULTS, MotionConfig, RecorderConfig
from ssc.features.dnn_person_detector import find_default_dnn_model_path
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
    parser = argparse.ArgumentParser(description="Run WiseNET test split with DNN settings.")
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("docs") / "project" / "wisenet_split.json",
        help="Split JSON path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / "wisenet_dnn",
        help="Base output directory for DNN runs.",
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

    dnn_model_path = find_default_dnn_model_path(Path(__file__).resolve().parents[1])
    if dnn_model_path is None:
        raise SystemExit("Default MobileNet-SSD model not found in models/.")

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
            person_detector="dnn",
            dnn_model_path=dnn_model_path,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

```

---

## run_wisenet_hog.py <a id='tools-run_wisenet_hog-py'></a>

```python
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

```

---

## run_yolo.py <a id='tools-run_yolo-py'></a>

```python
from __future__ import annotations

# This file was AI assisted as it is mostly for evaluations and testing purposes.

import argparse
import threading
import sys
from pathlib import Path


# This script was heavily AI-generated to provide a more intelligent evaluation runner.
# Reason: fast scaffolding for YOLO demo runs; reviewed and adjusted to match project defaults.


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

```

---

## validate_hog_integrity.py <a id='tools-validate_hog_integrity-py'></a>

```python
from __future__ import annotations

# This file was AI assisted as it is mostly for evaluations and testing purposes.

import argparse
from pathlib import Path

import cv2
import numpy as np


def _try_import_skimage():
    try:
        from skimage.feature import hog  # type: ignore

        return hog
    except Exception:
        return None


def _read_first_frame(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {video_path}")
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise SystemExit(f"Failed to read first frame: {video_path}")
    return frame


def _opencv_hog_descriptor_size() -> int:
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return int(hog.getDescriptorSize())


def _opencv_hog_compute(frame_bgr: np.ndarray) -> int:
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    win_w, win_h = 64, 128
    if gray.shape[1] < win_w or gray.shape[0] < win_h:
        gray = cv2.resize(gray, (max(win_w, gray.shape[1]), max(win_h, gray.shape[0])))

    patch = gray[0:win_h, 0:win_w]
    feats = hog.compute(patch)
    if feats is None:
        raise SystemExit("OpenCV HOG returned no features")
    return int(feats.shape[0])


def main() -> int:
    p = argparse.ArgumentParser(description="Validate OpenCV HOG integrity (dimension + visual inspection).")
    p.add_argument("--video", type=Path, required=True, help="Video file to sample first frame from")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("outputs") / "hog_validation",
        help="Output folder for any generated images",
    )
    args = p.parse_args()

    frame = _read_first_frame(args.video)

    expected = _opencv_hog_descriptor_size()
    got = _opencv_hog_compute(frame)
    print(f"OpenCV HOG descriptor size (expected): {expected}")
    print(f"OpenCV HOG computed feature length:     {got}")
    if expected != got:
        raise SystemExit("HOG dimensionality mismatch: descriptor size != computed feature length")

    hog_fn = _try_import_skimage()
    if hog_fn is None:
        print("scikit-image not installed; skipping HOG visualisation. Install: pip install scikit-image")
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (640, int(round(640 * (gray.shape[0] / max(1, gray.shape[1]))))))

    features, hog_image = hog_fn(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=True,
        feature_vector=True,
    )

    hog_image = hog_image - hog_image.min()
    if float(hog_image.max()) > 0:
        hog_image = hog_image / hog_image.max()
    hog_u8 = (hog_image * 255.0).astype(np.uint8)

    cv2.imwrite(str(out_dir / "hog_input_gray.png"), gray)
    cv2.imwrite(str(out_dir / "hog_visualisation.png"), hog_u8)
    print(f"Saved: {out_dir / 'hog_input_gray.png'}")
    print(f"Saved: {out_dir / 'hog_visualisation.png'}")
    print(f"scikit-image HOG feature length: {int(features.shape[0])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

```

---

## wisenet_class_balance.py <a id='tools-wisenet_class_balance-py'></a>

```python
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

```

---

## wisenet_eval.py <a id='tools-wisenet_eval-py'></a>

```python
from __future__ import annotations

# This file was AI assisted as it is mostly for evaluations and testing purposes.

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

```

---

## wisenet_split.py <a id='tools-wisenet_split-py'></a>

```python
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

```

---

