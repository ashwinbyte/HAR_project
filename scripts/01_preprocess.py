import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import os

DATA_PATH   = "../data/WISDM_ar_v1.1_raw.txt"
OUTPUT_DIR  = "../data/"
WINDOW_SIZE = 51
STEP_SIZE   = 26
TEST_SIZE   = 0.20
SEED        = 42
ACTIVITIES  = ['Walking','Jogging','Upstairs','Downstairs','Sitting','Standing']

print("Step 1: Loading raw data...")
lines = []
with open(DATA_PATH, 'r') as f:
    for line in f:
        line = line.strip().rstrip(';')
        if line:
            lines.append(line)

rows = []
for line in lines:
    parts = line.split(',')
    if len(parts) == 6:
        try:
            rows.append({
                'user_id':   parts[0].strip(),
                'activity':  parts[1].strip(),
                'timestamp': float(parts[2].strip()),
                'x':         float(parts[3].strip()),
                'y':         float(parts[4].strip()),
                'z':         float(parts[5].strip())
            })
        except:
            continue

df = pd.DataFrame(rows)
df = df[df['activity'].isin(ACTIVITIES)].reset_index(drop=True)
print(f"  Loaded {len(df):,} rows, {df['user_id'].nunique()} users")
print(f"  Activities:\n{df['activity'].value_counts().to_string()}")

print("\nStep 2: Segmenting into windows...")
X, y = [], []
for (user, activity), grp in df.groupby(['user_id', 'activity']):
    data = grp[['x','y','z']].values.astype(np.float32)
    for start in range(0, len(data) - WINDOW_SIZE, STEP_SIZE):
        window = data[start:start + WINDOW_SIZE]
        if len(window) == WINDOW_SIZE:
            X.append(window)
            y.append(activity)

X = np.array(X)
y = np.array(y)
print(f"  Total windows: {len(X):,}")
print(f"  Window shape:  {X.shape}")

print("\nStep 3: Z-score normalizing...")
X_norm = np.zeros_like(X)
for i in range(len(X)):
    for ax in range(3):
        col  = X[i, :, ax]
        mean = col.mean()
        std  = col.std()
        X_norm[i, :, ax] = (col - mean) / std if std > 1e-8 else col - mean
X = X_norm
print("  Done.")

print("\nStep 4: Encoding labels...")
le       = LabelEncoder()
y_int    = le.fit_transform(y)
y_onehot = to_categorical(y_int, num_classes=6)
print(f"  Classes: {list(le.classes_)}")

print("\nStep 5: Splitting 80/20...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=y_int
)
print(f"  X_train: {X_train.shape}")
print(f"  X_test:  {X_test.shape}")

print("\nStep 6: Saving files...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.save(f"{OUTPUT_DIR}X_train.npy",       X_train)
np.save(f"{OUTPUT_DIR}X_test.npy",        X_test)
np.save(f"{OUTPUT_DIR}y_train.npy",       y_train)
np.save(f"{OUTPUT_DIR}y_test.npy",        y_test)
np.save(f"{OUTPUT_DIR}label_classes.npy", le.classes_)
print("  Saved all files.")
print("\n✅ Preprocessing COMPLETE!")
