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
