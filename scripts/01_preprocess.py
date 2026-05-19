import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


DATA_PATH = "../data/WISDM_ar_v1.1_raw.txt"
OUT_DIR = "../data"

WINDOW_SIZE = 51
STEP_SIZE = 26
TEST_SIZE = 0.20
RANDOM_STATE = 42

activities = [
    "Walking",
    "Jogging",
    "Upstairs",
    "Downstairs",
    "Sitting",
    "Standing"
]


print("Reading dataset...")

records = []

with open(DATA_PATH, "r") as file:
    for line in file:
        line = line.strip().replace(";", "")

        if not line:
            continue

        parts = line.split(",")

        if len(parts) != 6:
            continue

        try:
            user_id = parts[0].strip()
            activity = parts[1].strip()
            timestamp = float(parts[2].strip())
            x = float(parts[3].strip())
            y = float(parts[4].strip())
            z = float(parts[5].strip())

            records.append([user_id, activity, timestamp, x, y, z])

        except ValueError:
            continue


df = pd.DataFrame(
    records,
    columns=["user_id", "activity", "timestamp", "x", "y", "z"]
)

df = df[df["activity"].isin(activities)].reset_index(drop=True)

print("Rows loaded:", len(df))
print("Users:", df["user_id"].nunique())
print(df["activity"].value_counts())


# create sliding windows
X = []
y = []

for (user_id, activity), group in df.groupby(["user_id", "activity"]):
    values = group[["x", "y", "z"]].values.astype("float32")

    for start in range(0, len(values) - WINDOW_SIZE, STEP_SIZE):
        end = start + WINDOW_SIZE
        window = values[start:end]

        if window.shape[0] == WINDOW_SIZE:
            X.append(window)
            y.append(activity)


X = np.array(X, dtype="float32")
y = np.array(y)

print("Windows created:", X.shape)


# normalize each window separately
for i in range(X.shape[0]):
    for j in range(X.shape[2]):
        mean = X[i, :, j].mean()
        std = X[i, :, j].std()

        if std != 0:
            X[i, :, j] = (X[i, :, j] - mean) / std
        else:
            X[i, :, j] = X[i, :, j] - mean


encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_cat = to_categorical(y_encoded, num_classes=len(encoder.classes_))

print("Labels:", list(encoder.classes_))


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_cat,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_encoded
)

print("Train data:", X_train.shape, y_train.shape)
print("Test data:", X_test.shape, y_test.shape)


os.makedirs(OUT_DIR, exist_ok=True)

np.save(os.path.join(OUT_DIR, "X_train.npy"), X_train)
np.save(os.path.join(OUT_DIR, "X_test.npy"), X_test)
np.save(os.path.join(OUT_DIR, "y_train.npy"), y_train)
np.save(os.path.join(OUT_DIR, "y_test.npy"), y_test)
np.save(os.path.join(OUT_DIR, "label_classes.npy"), encoder.classes_)

print("Preprocessing finished.")
