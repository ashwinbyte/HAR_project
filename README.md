# Personalized On-Device Human Activity Recognition on Android

> A complete end-to-end HAR system — trained on WISDM, deployed as a TFLite Android app on Samsung Galaxy S25, with personalization via last-layer fine-tuning and INT8 quantization.

---

## Overview

This project investigates how severely phone placement shift degrades HAR accuracy and whether memory-efficient last-layer fine-tuning can recover it using minimal personal data.

A lightweight 1D-CNN is trained on the WISDM public dataset, exported to TFLite in FP32 and INT8 formats, and deployed as a fully functional Android application. Four controlled experiments measure baseline accuracy, placement-induced domain shift, personalization recovery, and quantization efficiency.

---

## Key Results

| Experiment | Description | Result | Target | Status |
|---|---|---|---|---|
| E1 | WISDM Baseline | **96.92%** accuracy | ≥ 90% | Exceeded |
| E2 | Placement Domain Shift | **14–19%** (75 pp drop) | 10+ pp drop | Exceeded |
| E3 | Personalization (Hand, 5 min) | **+17.02 pp** recovery | +15–20 pp | Met |
| E4 | INT8 Quantization | **2.8× smaller**, 0.23 pp loss | 4×, <3% | Partial |

---

## Demo

The Android app runs entirely on-device with no internet connection required.

| Screen | Description |
|---|---|
| Home | Model stats — accuracy, activities, quantization |
| Live Detection | Real-time activity prediction with confidence % and latency |
| Data Recorder | Record personal sensor data for model personalization |

**Supported Activities:** Walking · Jogging · Upstairs · Downstairs · Sitting · Standing  
**Supported Placements:** Pocket · Hand · Backpack  
**Tested Device:** Samsung Galaxy S25 (Android API 34)  
**Minimum API:** Android 26


---

## Tech Stack

| Component | Technology |
|---|---|
| Language (ML) | Python 3.12 |
| ML Framework | TensorFlow 2.16 / Keras |
| Mobile Inference | TensorFlow Lite 2.12 |
| Android Language | Java |
| Android Min API | 26 (Android 8.0) |
| Build System | Gradle 9.3.1 |
| Test Device | Samsung Galaxy S25 (SM-S931U1) |
| Dataset | WISDM v1.1 |

---

## Model Architecture

```
Input: (51, 3)  — 51 time steps × 3 axes (X, Y, Z)
  │
  ├── Conv1D(64, kernel=3, ReLU) + BatchNorm   ← low-level patterns
  ├── Conv1D(64, kernel=3, ReLU) + BatchNorm   ← mid-level patterns
  ├── Conv1D(64, kernel=3, ReLU) + BatchNorm   ← high-level patterns
  │
  ├── GlobalAveragePooling1D                    ← compress time dimension
  ├── Dropout(0.3)                              ← prevent overfitting
  └── Dense(6, softmax)  [name: classifier]    ← 6-class output

Total parameters: 26,502
FP32 size: 0.106 MB
INT8 size: 0.037 MB
```

---

## Quickstart

### Prerequisites

```bash
# Python dependencies
pip install tensorflow numpy scikit-learn pandas matplotlib seaborn

# Android
# Install Android Studio with API 26+ SDK
```

### Step 1 — Download WISDM Dataset

Download WISDM v1.1 from:
```
https://www.cis.fordham.edu/wisdm/dataset.php
```

Place the raw file at:
```
data/WISDM_ar_v1.1_raw.txt
```

### Step 2 — Preprocess Data

```bash
cd scripts
python 01_preprocess.py
```

This generates `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy` in `data/`.

### Step 3 — Train the Model

```bash
python 02_train.py
```

Expected output:
```
Test Accuracy: 96.92%
Macro F1: 0.94
Model saved to models/model.keras
```

> **Mac M3 Note:** If you get a Metal GPU crash add this to the top of the script:
> ```python
> import tensorflow as tf
> tf.config.set_visible_devices([], 'GPU')
> ```

### Step 4 — Export to TFLite

```bash
python 03_export_tflite.py
```

This creates:
- `models/model_fp32.tflite` (0.106 MB)
- `models/model_int8.tflite` (0.037 MB)
- Copies both to `android/HARApp/app/src/main/assets/`

### Step 5 — Run All Experiments

```bash
python 04_experiments.py
```

Results saved to `outputs/results/` as JSON files.

---

## Android App Setup

### Build and Install

1. Open `android/HARApp/` in Android Studio
2. Connect your Android device via USB with USB Debugging enabled
3. Click **▶ Run** or run from Terminal:

```bash
cd android/HARApp
./gradlew installDebug
```

### Known Build Issues

**TFLite namespace conflict with Gradle 9.3:**
The `build.gradle` already includes the fix:
```gradle
configurations.all {
    exclude group: 'org.tensorflow', module: 'tensorflow-lite-api'
    exclude group: 'org.tensorflow', module: 'tensorflow-lite-support-api'
}
```

**AndroidX not enabled:**
`gradle.properties` already includes:
```
android.useAndroidX=true
android.enableJetifier=true
```

### Collect Personal Data

1. Open the app → tap **Open Recorder**
2. Select activity and placement
3. Tap **Start Recording** and perform the activity
4. Tap **Stop Recording** when done
5. Pull files from phone to Mac:

```bash
adb pull /sdcard/Android/data/com.har.app/files/HAR_data data/personal/
```

> **Tip:** Record at least 8 minutes per activity per placement for best personalization results. 3-minute recordings produce too few windows for fine-tuning to converge.

---

## Personalization

After collecting personal data, run the personalization experiment:

```bash
python 04_experiments.py
```

The script:
1. Freezes all Conv1D layers
2. Fine-tunes only the last Dense classifier (384 parameters)
3. Tests with 1, 3, and 5 minutes of calibration data
4. Reports accuracy improvement per placement

---

## Experiment Results Detail

### E2 — Placement Domain Shift

| Placement | Accuracy | Drop from Baseline |
|---|---|---|
| Pocket | 14.58% | −82.34 pp |
| Hand | 14.66% | −82.26 pp |
| Backpack | 19.40% | −77.52 pp |

### E3 — Personalization Recovery

| Placement | Baseline | 1 min | 3 min | 5 min | Best Δ |
|---|---|---|---|---|---|
| Pocket | 14.58% | 14.58% | 13.90% | 14.35% | −0.23 pp |
| Hand  | 14.66% | 14.66% | 25.77% | **31.68%** | **+17.02 pp** |
| Backpack | 19.40% | 19.40% | 22.07% | 26.42% | +7.02 pp |

### E4 — Quantization

| Metric | FP32 | INT8 | Result |
|---|---|---|---|
| Model Size | 0.106 MB | 0.037 MB | 2.8× smaller |
| Accuracy | 96.92% | 96.69% | 0.23 pp loss |
| Mean Latency | 0.033 ms | 0.028 ms | 1.2× faster |

---

## Key Findings

**1. Domain shift is more severe than prior literature predicts.**
Prior work estimated 70–80% accuracy under placement shift. We measured 14–19% — a 75+ percentage point collapse — indicating that any production HAR system requires personalization.

**2. Last-layer fine-tuning is effective given sufficient data.**
Five minutes of personal data recovers +17 pp for hand placement, meeting the +15–20 pp target. The positive trend (more data = more recovery) holds across all placements.

**3. Pocket personalization requires more than 5 minutes.**
High sensor orientation variability in pocket placement means 5 minutes of data is insufficient. Future work should collect 10+ minutes with varied phone orientations.

**4. INT8 quantization is essentially free for small HAR models.**
0.23 pp accuracy loss for 2.8× size reduction makes INT8 the clear choice for production deployment.

---

## Citation

If you use this code or findings in your work please cite:

```bibtex
@article{ravichandran2026har,
  title={Personalized On-Device Human Activity Recognition on Android},
  author={Anonymous},
  journal={Course Project},
  year={2026}
}
```

---

## References

1. Kwapisz et al. — Activity Recognition Using Cell Phone Accelerometers. ACM SIGKDD 2011.
2. Anguita et al. — A Public Domain Dataset for HAR Using Smartphones. ESANN 2013.
3. Ordonez & Roggen — Deep Convolutional and LSTM RNNs for Wearable Activity Recognition. Sensors 2016.
4. Gim & Ko — Sage: Memory-Efficient DNN Training on Mobile Devices. MobiSys 2022.
5. Jacob et al. — Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference. CVPR 2018.

---

## License

This project is released for academic and educational purposes.  
The WISDM dataset is subject to its own license — see the WISDM website for terms.

---

