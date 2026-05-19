import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import time
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report


try:
    tf.config.set_visible_devices([], "GPU")
except:
    pass


MODEL_PATH = "../models/model.keras"
DATA_DIR = "../data"
PERSONAL_DIR = "../data/personal"
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


def read_personal_file(path):
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


def get_placement_data(placement, split_ratio=0.7):
    files = []

    for name in os.listdir(PERSONAL_DIR):
        if name.startswith(placement + "_") and name.endswith(".csv"):
            files.append(name)

    X_cal_parts = []
    y_cal_parts = []
    X_test_parts = []
    y_test_parts = []

    for name in files:
        file_path = os.path.join(PERSONAL_DIR, name)
        X, y = read_personal_file(file_path)

        if len(X) < 5:
            continue

        split_idx = max(1, int(len(X) * split_ratio))

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
model = tf.keras.models.load_model(MODEL_PATH)


# Experiment 2: test the original model on each phone placement
print("\nExperiment 2: placement test without fine-tuning")

exp2_results = {}

for placement in PLACEMENTS:
    _, _, X_test, y_test = get_placement_data(placement)

    if len(X_test) == 0:
        print(placement, "has no test data")
        continue

    y_true = label_encoder.transform(y_test)

    pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(pred_probs, axis=1)

    acc = accuracy_score(y_true, y_pred)
    exp2_results[placement] = round(acc * 100, 2)

    print("\nPlacement:", placement)
    print("Accuracy:", round(acc * 100, 2))

    present_labels = sorted(set(y_true))
    present_names = [classes[i] for i in present_labels]

    print(
        classification_report(
            y_true,
            y_pred,
            labels=present_labels,
            target_names=present_names,
            zero_division=0
        )
    )


with open(os.path.join(RESULTS_DIR, "exp2_domainshift.json"), "w") as f:
    json.dump(exp2_results, f, indent=2)


# Experiment 3: fine-tune classifier layer with small personal data
print("\nExperiment 3: personalization")

minutes_list = [1, 3, 5]
windows_per_min = 15

exp3_results = {}

for placement in PLACEMENTS:
    X_cal, y_cal, X_test, y_test = get_placement_data(placement)

    if len(X_test) == 0:
        continue

    y_test_int = label_encoder.transform(y_test)

    placement_scores = {}

    print("\nPlacement:", placement)
    print("Calibration windows:", len(X_cal))
    print("Test windows:", len(X_test))

    for minutes in minutes_list:
        tuned_model = tf.keras.models.load_model(MODEL_PATH)

        for layer in tuned_model.layers:
            layer.trainable = False

        # train only the last layer
        tuned_model.layers[-1].trainable = True

        tuned_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        n = min(minutes * windows_per_min, len(X_cal))

        X_sub = X_cal[:n]
        y_sub = label_encoder.transform(y_cal[:n])
        y_sub = tf.keras.utils.to_categorical(y_sub, num_classes=len(classes))

        tuned_model.fit(
            X_sub,
            y_sub,
            epochs=20,
            batch_size=16,
            verbose=0
        )

        pred_probs = tuned_model.predict(X_test, verbose=0)
        y_pred = np.argmax(pred_probs, axis=1)

        acc = accuracy_score(y_test_int, y_pred)
        placement_scores[str(minutes) + "min"] = round(acc * 100, 2)

        print(minutes, "min:", round(acc * 100, 2))

    exp3_results[placement] = placement_scores


print("\nPersonalization results")
print(f"{'Placement':<12} {'1 min':>8} {'3 min':>8} {'5 min':>8}")
print("-" * 40)

for placement, scores in exp3_results.items():
    print(
        f"{placement:<12} "
        f"{str(scores.get('1min', '-')):>8} "
        f"{str(scores.get('3min', '-')):>8} "
        f"{str(scores.get('5min', '-')):>8}"
    )


with open(os.path.join(RESULTS_DIR, "exp3_personalization.json"), "w") as f:
    json.dump(exp3_results, f, indent=2)


# Experiment 4: compare FP32 and INT8 TFLite models
print("\nExperiment 4: TFLite model comparison")

X_test_wisdm = np.load(os.path.join(DATA_DIR, "X_test.npy")).astype("float32")
y_test_wisdm = np.argmax(np.load(os.path.join(DATA_DIR, "y_test.npy")), axis=1)


def test_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    size_mb = os.path.getsize(model_path) / (1024 * 1024)

    predictions = []
    times = []

    for i in range(len(X_test_wisdm)):
        sample = X_test_wisdm[i:i + 1]

        interpreter.set_tensor(input_details[0]["index"], sample)

        start = time.perf_counter()
        interpreter.invoke()
        elapsed = (time.perf_counter() - start) * 1000

        output = interpreter.get_tensor(output_details[0]["index"])

        predictions.append(np.argmax(output))
        times.append(elapsed)

    times = np.array(times)
    acc = accuracy_score(y_test_wisdm, predictions)

    return {
        "size_mb": round(float(size_mb), 4),
        "accuracy": round(float(acc) * 100, 2),
        "latency_mean_ms": round(float(times.mean()), 3),
        "latency_p95_ms": round(float(np.percentile(times, 95)), 3)
    }


exp4_results = {}

models_to_test = {
    "FP32": "../models/model_fp32.tflite",
    "INT8": "../models/model_int8.tflite"
}

for name, path in models_to_test.items():
    if not os.path.exists(path):
        print(name, "model not found")
        continue

    print("\nTesting", name)

    result = test_tflite_model(path)
    exp4_results[name] = result

    print("Size:", result["size_mb"], "MB")
    print("Accuracy:", result["accuracy"], "%")
    print("Mean latency:", result["latency_mean_ms"], "ms")
    print("P95 latency:", result["latency_p95_ms"], "ms")


if "FP32" in exp4_results and "INT8" in exp4_results:
    fp32 = exp4_results["FP32"]
    int8 = exp4_results["INT8"]

    print("\nComparison")
    print("Size reduction:", round(fp32["size_mb"] / int8["size_mb"], 2), "x")
    print("Speed improvement:", round(fp32["latency_mean_ms"] / int8["latency_mean_ms"], 2), "x")
    print("Accuracy drop:", round(fp32["accuracy"] - int8["accuracy"], 2), "percentage points")


with open(os.path.join(RESULTS_DIR, "exp4_quantization.json"), "w") as f:
    json.dump(exp4_results, f, indent=2)


print("\nAll experiments finished.")
print("Results saved in:", RESULTS_DIR)
