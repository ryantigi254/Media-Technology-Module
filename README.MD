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

## 📂 Dataset Setup

The WiseNET dataset used for evaluation is too large to conduct version control on. You must download it locally.

**Option A: Automatic Download (Recommended)**
Use the included helper script to download and unzip the dataset to `data/wisenet`:
```bash
python tools/wisenet_split.py --download
```

**Option B: Manual Download**
Download `abdelrhmannile/wisenet` from Kaggle and unzip it into `data/wisenet`.
```bash
kaggle datasets download -d abdelrhmannile/wisenet -p data/wisenet --unzip
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
