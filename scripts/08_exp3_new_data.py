import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


try:
    tf.config.set_visible_devices([], "GPU")
except:
    pass


MODEL_PATH = "../models/model.keras"
DATA_DIR = "../data"
NEW_DIR = "../data/personal_new"
OLD_DIR = "../data/personal"
RESULTS_DIR = "../outputs/results"

WINDOW_SIZE = 51
STEP_SIZE = 26

ACTIVITIES = [
    "Walking",
    "Jogging",
    "Upstairs",
    "Downstairs",
    "Sitting",
    "Standing"
]

PLACEMENTS = ["pocket", "hand", "backpack"]

os.makedirs(RESULTS_DIR, exist_ok=True)


classes = np.load(os.path.join(DATA_DIR, "label_classes.npy"), allow_pickle=True)

label_encoder = LabelEncoder()
label_encoder.classes_ = classes


def load_activity_file(path):
    df = pd.read_csv(path)
    df = df[df["label"].isin(ACTIVITIES)].dropna()

    X = []
    y = []

    for activity, group in df.groupby("label"):
        values = group[["ax", "ay", "az"]].values.astype("float32")

        for start in range(0, len(values) - WINDOW_SIZE, STEP_SIZE):
            window = values[start:start + WINDOW_SIZE].copy()

            if window.shape[0] != WINDOW_SIZE:
                continue

            for axis in range(3):
                col = window[:, axis]
                mean = col.mean()
                std = col.std()

                if std > 1e-8:
                    window[:, axis] = (col - mean) / std
                else:
                    window[:, axis] = col - mean

            X.append(window)
            y.append(activity)

    return np.array(X), np.array(y)


def get_files_for_placement(placement):
    all_files = []

    for folder in [NEW_DIR, OLD_DIR]:
        if not os.path.exists(folder):
            continue

        for name in os.listdir(folder):
            if name.startswith(placement + "_") and name.endswith(".csv"):
                all_files.append(os.path.join(folder, name))

    best_file = {}

    for path in all_files:
        name = os.path.basename(path)
        parts = name.split("_")

        if len(parts) < 2:
            continue

        activity = parts[1]

        if activity not in ACTIVITIES:
            continue

        if activity not in best_file:
            best_file[activity] = path
        elif os.path.getsize(path) > os.path.getsize(best_file[activity]):
            best_file[activity] = path

    return best_file


def load_placement_data(placement):
    files = get_files_for_placement(placement)

    if len(files) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    print("\nFiles for", placement)

    X_cal_parts = []
    y_cal_parts = []
    X_test_parts = []
    y_test_parts = []

    for activity, path in files.items():
        try:
            rows = sum(1 for _ in open(path)) - 1
        except:
            rows = 0

        print(activity + ":", os.path.basename(path), "-", rows, "rows")

        X, y = load_activity_file(path)

        if len(X) < 10:
            continue

        split_idx = max(5, int(len(X) * 0.7))

        X_cal_parts.append(X[:split_idx])
        y_cal_parts.append(y[:split_idx])

        X_test_parts.append(X[split_idx:])
        y_test_parts.append(y[split_idx:])

    if len(X_cal_parts) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    X_cal = np.concatenate(X_cal_parts)
    y_cal = np.concatenate(y_cal_parts)

    X_test = np.concatenate(X_test_parts)
    y_test = np.concatenate(y_test_parts)

    return X_cal, y_cal, X_test, y_test


print("Loading model...")
base_model = tf.keras.models.load_model(MODEL_PATH)


print("\nBaseline results")

baseline_results = {}

for placement in PLACEMENTS:
    _, _, X_test, y_test = load_placement_data(placement)

    if len(X_test) == 0:
        print(placement, "has no test data")
        continue

    y_true = label_encoder.transform(y_test)

    pred_probs = base_model.predict(X_test, verbose=0)
    y_pred = np.argmax(pred_probs, axis=1)

    acc = accuracy_score(y_true, y_pred)
    baseline_results[placement] = round(acc * 100, 2)

    print(placement + ":", round(acc * 100, 2), "%")


print("\nPersonalization results")

windows_per_min = 46
duration_list = [1, 3, 5]

personalization_results = {}

for placement in PLACEMENTS:
    X_cal, y_cal, X_test, y_test = load_placement_data(placement)

    if len(X_test) == 0:
        continue

    y_test_int = label_encoder.transform(y_test)
    baseline_acc = baseline_results.get(placement, 0)

    scores = {}

    print("\nPlacement:", placement)
    print("Baseline:", baseline_acc)
    print("Calibration windows:", len(X_cal))
    print("Test windows:", len(X_test))

    for minutes in duration_list:
        model = tf.keras.models.load_model(MODEL_PATH)

        for layer in model.layers:
            layer.trainable = False

        # train only the final classifier layer
        model.layers[-1].trainable = True

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        n = min(minutes * windows_per_min, len(X_cal))

        X_sub = X_cal[:n]
        y_sub = label_encoder.transform(y_cal[:n])
        y_sub = tf.keras.utils.to_categorical(y_sub, num_classes=len(classes))

        stop_early = tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=8,
            restore_best_weights=True
        )

        model.fit(
            X_sub,
            y_sub,
            epochs=50,
            batch_size=32,
            callbacks=[stop_early],
            verbose=0
        )

        pred_probs = model.predict(X_test, verbose=0)
        y_pred = np.argmax(pred_probs, axis=1)

        acc = accuracy_score(y_test_int, y_pred)
        acc_percent = round(acc * 100, 2)
        improvement = round(acc_percent - baseline_acc, 2)

        scores[str(minutes) + "min"] = acc_percent

        print(minutes, "min:", acc_percent, "%", "change:", improvement, "pp")

    personalization_results[placement] = scores


print("\nFinal summary")
print(f"{'Placement':<12} {'Baseline':>10} {'1 min':>8} {'3 min':>8} {'5 min':>8}")
print("-" * 52)

for placement, scores in personalization_results.items():
    base = baseline_results.get(placement, "-")

    print(
        f"{placement:<12} "
        f"{str(base):>10} "
        f"{str(scores.get('1min', '-')):>8} "
        f"{str(scores.get('3min', '-')):>8} "
        f"{str(scores.get('5min', '-')):>8}"
    )


out_path = os.path.join(RESULTS_DIR, "exp3_personalization.json")

with open(out_path, "w") as f:
    json.dump(personalization_results, f, indent=2)


print("\nExperiment finished.")
print("Saved:", out_path)
