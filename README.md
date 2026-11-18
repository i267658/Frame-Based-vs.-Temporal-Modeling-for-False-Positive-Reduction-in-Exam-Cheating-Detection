# Frame-Based vs. Temporal Modeling for False Positive Reduction in Gaze-Based Exam Cheating Detection

A controlled empirical study comparing single-frame classification (**MLP**) against sequence-based temporal modeling (**LSTM**) for automated webcam exam proctoring, evaluated at a fixed **90% recall** target.

---

##  Project Overview & What Happens in This Study

Online exam proctoring systems often trigger false alarms when students naturally glance away while reading or thinking. While temporal models are commonly assumed to reduce false positives by considering recent behavioral context, existing literature rarely isolates temporal modeling while holding features, splits, and target recall constant.

**What we do in this repository:**
1. **Controlled Architecture Comparison:** We evaluate a **Frame MLP** (which sees only the current row, 3,137 parameters) against a **Temporal LSTM** (which processes consecutive windows of $W \in \{10, 30, 60\}$ rows, 22,593 parameters).
2. **Leakage-Free Preprocessing & Windowing:** We use a chronological 60% / 20% / 20% split (train / validation / test). Missing-value median imputation and feature scaling are fit strictly on the training set, and sliding windows are built independently inside each split.
3. **Threshold Calibration at 90% Recall:** Decision thresholds are calibrated on the validation split to achieve $\ge 90\%$ cheating recall and then frozen for unbiased evaluation on the test set.
4. **Edge Deployment & Quantization Analysis:** Both models are converted to **TensorFlow Lite (TFLite)** across FP32, dynamic-range INT8, and full-integer INT8 variants to benchmark latency and memory footprint.

---

## Key Findings

* **Lower False Positive Rate with Frame MLP:** Across all window lengths, the single-frame MLP achieved a lower False Positive Rate (FPR) than the temporal LSTM (**33.3% vs. 68.5%** at $W = 30$) and higher AUC (**~0.84 vs. ~0.77**).
* **Efficiency & Latency:** The Frame MLP requires ~7x fewer parameters (3,137 vs. 22,593), ~200x fewer FLOPs, and runs in **0.007 ms/sample** (TFLite FP32/INT8) compared to **0.103–0.184 ms/sample** for the LSTM.
* **Practical Takeaway:** On exploratory row-order windows without session timestamps, adding temporal sequence modeling increases model complexity without reducing false alarms, and FPR remains too high (33%–69%) for fully automated student flagging without human review.

---

##  Dataset

* **Included File:** `[your_dataset.csv]` (uploaded directly in this repository for immediate reproducibility)
* **Source:** *Students' Suspicious Behaviors Detection Dataset for AI-Powered Online Exam Proctoring* (Hossen & Uddin, Mendeley Data, DOI: `10.17632/39xs8th543.1`)
* **Size & Balance:** 5,500 labeled webcam behavior records (2,619 cheating / 2,881 non-cheating), captured at 720p (20 fps) from 30+ participants with no raw images or personal identifiers.
* **Selected Features (15 Numeric Columns):** To isolate gaze and head-pose dynamics without leaking object-detector shortcuts (such as phone presence), we retain 15 numeric features covering **head pose** (pitch, yaw, roll), **eye gaze coordinates**, and **eye/nose landmarks**.

---

##  Repository Contents

* `[your_notebook.ipynb]` — Complete reproducible Jupyter Notebook covering preprocessing, windowing, MLP & LSTM training (across seeds `0, 1, 2`), threshold freezing, TFLite quantization, and evaluation plots.
* `[your_dataset.csv]` — Dataset of 5,500 webcam proctoring records used for training, validation, and testing.
* `[your_recording.mp4]` — Screen recording walkthrough demonstrating the notebook execution and experimental results.
* `README.md` — Project overview, methodology summary, key findings, and repository documentation.

---

##  Authors

* **Mona Khalil** — Department of Data Science and Artificial Intelligence, Islamabad, Pakistan
* **Ezza Razaq** — Department of Data Science and Artificial Intelligence, Islamabad, Pakistan
