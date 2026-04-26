import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import pandas as pd
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import json

MODEL_PATH   = "../models/model.keras"
NEW_DIR      = "../data/personal_new/"
OLD_DIR      = "../data/personal/"
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

def load_all_for_placement(placement):
    """Load ALL files for a placement from both old and new directories.
    Use longest file as calibration source."""
    all_files = []
    for d in [NEW_DIR, OLD_DIR]:
        files = [os.path.join(d, f) for f in os.listdir(d)
                 if f.startswith(placement+'_') and f.endswith('.csv')]
        all_files.extend(files)

    # Group by activity, keep longest file per activity
    act_files = {}
    for f in all_files:
        # Get activity from filename
        basename = os.path.basename(f)
        parts    = basename.split('_')
        if len(parts) < 2:
            continue
        activity = parts[1]
        if activity not in ACTIVITIES:
            continue
        size = os.path.getsize(f)
        if activity not in act_files or size > os.path.getsize(act_files[activity]):
            act_files[activity] = f

    print(f"  Files used for {placement}:")
    for act, f in act_files.items():
        lines = sum(1 for _ in open(f)) - 1
        print(f"    {act}: {os.path.basename(f)} ({lines} rows)")

    X_cal_all,  y_cal_all  = [], []
    X_test_all, y_test_all = [], []

    for activity, f in act_files.items():
        X, y = load_csv(f)
        if len(X) < 10:
            continue
        split = max(5, int(len(X) * 0.7))
        X_cal_all.append(X[:split])
        y_cal_all.append(y[:split])
        X_test_all.append(X[split:])
        y_test_all.append(y[split:])

    if not X_cal_all:
        return np.array([]),np.array([]),np.array([]),np.array([])

    return (np.concatenate(X_cal_all), np.concatenate(y_cal_all),
            np.concatenate(X_test_all), np.concatenate(y_test_all))

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# ── Baseline per placement ────────────────────────────────────────────────────
print("\n── Baseline (no fine-tuning) ────────────────────────")
exp2_results = {}
for placement in PLACEMENTS:
    _, _, X_test, y_test = load_all_for_placement(placement)
    if len(X_test) == 0:
        continue
    y_int  = le.transform(y_test)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc    = accuracy_score(y_int, y_pred)
    exp2_results[placement] = round(acc*100, 2)
    print(f"  {placement:10s}: {acc*100:.2f}%")

# ── Experiment 3 ──────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("EXPERIMENT 3 — Personalization Recovery")
print("="*55)

WINDOWS_PER_MIN = 46
DURATIONS       = [1, 3, 5]
exp3_results    = {}

for placement in PLACEMENTS:
    X_cal, y_cal, X_test, y_test = load_all_for_placement(placement)
    if len(X_test) == 0:
        continue

    y_test_int        = le.transform(y_test)
    baseline_acc      = exp2_results.get(placement, 0)
    placement_results = {}

    print(f"\n  Placement: {placement.upper()}")
    print(f"  Baseline: {baseline_acc:.2f}%")
    print(f"  Cal windows: {len(X_cal)}  Test windows: {len(X_test)}")

    for minutes in DURATIONS:
        m = tf.keras.models.load_model(MODEL_PATH)
        for layer in m.layers:
            layer.trainable = False
        m.get_layer('classifier').trainable = True
        m.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        n     = min(minutes * WINDOWS_PER_MIN, len(X_cal))
        X_sub = X_cal[:n]
        y_sub = le.transform(y_cal[:n])
        y_oh  = tf.keras.utils.to_categorical(y_sub, 6)

        cb = tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=8,
            restore_best_weights=True
        )
        m.fit(X_sub, y_oh,
              epochs=50,
              batch_size=32,
              callbacks=[cb],
              verbose=0)

        y_pred      = np.argmax(m.predict(X_test, verbose=0), axis=1)
        acc         = accuracy_score(y_test_int, y_pred)
        improvement = acc*100 - baseline_acc
        placement_results[f"{minutes}min"] = round(acc*100, 2)
        print(f"    {minutes} min → {acc*100:.2f}%  "
              f"(improvement: {improvement:+.2f} pp)")

    exp3_results[placement] = placement_results

print("\n── Exp 3 Final Summary ──────────────────────────────")
print(f"{'Placement':<12} {'Baseline':>10} {'1 min':>8} "
      f"{'3 min':>8} {'5 min':>8}")
print("-"*52)
for p, r in exp3_results.items():
    base = exp2_results.get(p, '-')
    print(f"{p:<12} {str(base):>10} "
          f"{str(r.get('1min','-')):>8} "
          f"{str(r.get('3min','-')):>8} "
          f"{str(r.get('5min','-')):>8}")

with open(f"{OUTPUT_DIR}exp3_personalization.json", 'w') as f:
    json.dump(exp3_results, f, indent=2)
print("\n✅ Experiment 3 COMPLETE!")
