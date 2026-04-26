import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import pandas as pd
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import json

MODEL_PATH   = "../models/model.keras"
PERSONAL_DIR = "../data/personal/"
OUTPUT_DIR   = "../outputs/results/"
WINDOW_SIZE  = 51
STEP_SIZE    = 26
ACTIVITIES   = ['Walking','Jogging','Upstairs','Downstairs','Sitting','Standing']
PLACEMENTS   = ['pocket','hand','backpack']
os.makedirs(OUTPUT_DIR, exist_ok=True)

classes = np.load("../data/label_classes.npy", allow_pickle=True)
le = LabelEncoder()
le.classes_ = classes

def load_csv(path):
    df = pd.read_csv(path)
    df = df[df['label'].isin(ACTIVITIES)].dropna()
    X, y = [], []
    for act, grp in df.groupby('label'):
        data = grp[['ax','ay','az']].values.astype(np.float32)
        for start in range(0, len(data) - WINDOW_SIZE, STEP_SIZE):
            win = data[start:start+WINDOW_SIZE].copy()
            if len(win) == WINDOW_SIZE:
                for ax in range(3):
                    col = win[:, ax]
                    std = col.std()
                    win[:, ax] = (col-col.mean())/std if std>1e-8 else col-col.mean()
                X.append(win)
                y.append(act)
    return np.array(X), np.array(y)

def load_placement_split(placement, cal_ratio=0.7):
    """Load all files for a placement.
    For each activity file: first cal_ratio = calibration, rest = test.
    This ensures ALL activities appear in both calibration and test sets."""
    files = [f for f in os.listdir(PERSONAL_DIR)
             if f.startswith(placement+'_') and f.endswith('.csv')]

    X_cal_all,  y_cal_all  = [], []
    X_test_all, y_test_all = [], []

    for f in files:
        X, y = load_csv(os.path.join(PERSONAL_DIR, f))
        if len(X) < 5:
            continue
        split = max(1, int(len(X) * cal_ratio))
        X_cal_all.append(X[:split])
        y_cal_all.append(y[:split])
        X_test_all.append(X[split:])
        y_test_all.append(y[split:])

    if not X_cal_all:
        return np.array([]),np.array([]),np.array([]),np.array([])

    X_cal  = np.concatenate(X_cal_all)
    y_cal  = np.concatenate(y_cal_all)
    X_test = np.concatenate(X_test_all)
    y_test = np.concatenate(y_test_all)
    return X_cal, y_cal, X_test, y_test

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")

# ── EXPERIMENT 2 ──────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("EXPERIMENT 2 — Placement Domain Shift (NO fine-tuning)")
print("="*55)
exp2_results = {}

for placement in PLACEMENTS:
    _, _, X_test, y_test = load_placement_split(placement)
    if len(X_test) == 0:
        print(f"  {placement}: no data")
        continue
    y_int  = le.transform(y_test)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc    = accuracy_score(y_int, y_pred)
    exp2_results[placement] = round(acc*100, 2)
    present = sorted(set(y_int))
    names   = [list(classes)[i] for i in present]
    print(f"\n  {placement.upper()}: {acc*100:.2f}%")
    print(classification_report(y_int, y_pred,
        labels=present, target_names=names, zero_division=0))

print("\n── Exp 2 Summary ────────────────────────────────────")
for p, a in exp2_results.items():
    print(f"  {p:10s}: {a:.2f}%")
with open(f"{OUTPUT_DIR}exp2_domainshift.json", 'w') as f:
    json.dump(exp2_results, f, indent=2)

# ── EXPERIMENT 3 ──────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("EXPERIMENT 3 — Personalization Recovery")
print("="*55)
DURATIONS       = [1, 3, 5]
WINDOWS_PER_MIN = 15
exp3_results    = {}

for placement in PLACEMENTS:
    X_cal, y_cal, X_test, y_test = load_placement_split(placement)
    if len(X_test) == 0:
        continue
    y_test_int       = le.transform(y_test)
    placement_results = {}
    print(f"\n  Placement: {placement.upper()}")
    print(f"  Cal windows: {len(X_cal)}  Test windows: {len(X_test)}")

    for minutes in DURATIONS:
        m = tf.keras.models.load_model(MODEL_PATH)
        for layer in m.layers:
            layer.trainable = False
        m.get_layer('classifier').trainable = True
        m.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        n     = min(minutes * WINDOWS_PER_MIN, len(X_cal))
        X_sub = X_cal[:n]
        y_sub = le.transform(y_cal[:n])
        y_oh  = tf.keras.utils.to_categorical(y_sub, 6)
        m.fit(X_sub, y_oh, epochs=20, batch_size=16, verbose=0)
        y_pred = np.argmax(m.predict(X_test, verbose=0), axis=1)
        acc    = accuracy_score(y_test_int, y_pred)
        placement_results[f"{minutes}min"] = round(acc*100, 2)
        print(f"    {minutes} min → {acc*100:.2f}%")

    exp3_results[placement] = placement_results

print("\n── Exp 3 Summary ────────────────────────────────────")
print(f"{'Placement':<12} {'1 min':>8} {'3 min':>8} {'5 min':>8}")
print("-"*40)
for p, r in exp3_results.items():
    print(f"{p:<12} {str(r.get('1min','-')):>8} "
          f"{str(r.get('3min','-')):>8} {str(r.get('5min','-')):>8}")
with open(f"{OUTPUT_DIR}exp3_personalization.json", 'w') as f:
    json.dump(exp3_results, f, indent=2)

# ── EXPERIMENT 4 ──────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("EXPERIMENT 4 — Quantization Efficiency")
print("="*55)
import time
X_test_wisdm = np.load("../data/X_test.npy").astype(np.float32)
y_test_wisdm = np.argmax(np.load("../data/y_test.npy"), axis=1)

def eval_tflite(path):
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    inp     = interp.get_input_details()
    out     = interp.get_output_details()
    size_mb = os.path.getsize(path) / (1024*1024)
    preds, lats = [], []
    for i in range(len(X_test_wisdm)):
        s = X_test_wisdm[i:i+1]
        interp.set_tensor(inp[0]['index'], s)
        t0 = time.perf_counter()
        interp.invoke()
        lats.append((time.perf_counter()-t0)*1000)
        preds.append(np.argmax(interp.get_tensor(out[0]['index'])))
    acc  = accuracy_score(y_test_wisdm, preds)
    lats = np.array(lats)
    return {
        'size_mb':         round(size_mb, 4),
        'accuracy':        round(acc*100, 2),
        'latency_mean_ms': round(lats.mean(), 3),
        'latency_p95_ms':  round(np.percentile(lats, 95), 3)
    }

exp4 = {}
for name, path in [('FP32','../models/model_fp32.tflite'),
                   ('INT8','../models/model_int8.tflite')]:
    if not os.path.exists(path):
        print(f"  {name}: not found")
        continue
    print(f"\n  Evaluating {name}...")
    r = eval_tflite(path)
    exp4[name] = r
    print(f"  Size:     {r['size_mb']} MB")
    print(f"  Accuracy: {r['accuracy']}%")
    print(f"  Latency:  {r['latency_mean_ms']} ms (mean)  "
          f"{r['latency_p95_ms']} ms (p95)")

if 'FP32' in exp4 and 'INT8' in exp4:
    print(f"\n  Size reduction:    "
          f"{exp4['FP32']['size_mb']/exp4['INT8']['size_mb']:.1f}x smaller")
    print(f"  Speed improvement: "
          f"{exp4['FP32']['latency_mean_ms']/exp4['INT8']['latency_mean_ms']:.1f}x faster")
    print(f"  Accuracy drop:     "
          f"{exp4['FP32']['accuracy']-exp4['INT8']['accuracy']:.2f} pp")

with open(f"{OUTPUT_DIR}exp4_quantization.json", 'w') as f:
    json.dump(exp4, f, indent=2)

print("\n✅ ALL EXPERIMENTS COMPLETE!")
print(f"Results saved to {OUTPUT_DIR}")
