
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
