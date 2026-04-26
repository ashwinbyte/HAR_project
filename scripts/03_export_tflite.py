import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import shutil, json

MODEL_DIR   = "../models/"
DATA_DIR    = "../data/"
ANDROID_DIR = "../android/assets/"
RES_DIR     = "../outputs/results/"
os.makedirs(ANDROID_DIR, exist_ok=True)
os.makedirs(RES_DIR,     exist_ok=True)

print("Loading model...")
model = tf.keras.models.load_model(f"{MODEL_DIR}model.keras")
print("Model loaded.")

# ── FP32 export using concrete function ───────────────────────────────────────
print("\nExporting FP32...")
run_model = tf.function(lambda x: model(x))
concrete  = run_model.get_concrete_function(
    tf.TensorSpec(shape=[1, 51, 3], dtype=tf.float32)
)
converter   = tf.lite.TFLiteConverter.from_concrete_functions([concrete], model)
tflite_fp32 = converter.convert()
fp32_path   = f"{MODEL_DIR}model_fp32.tflite"
with open(fp32_path, 'wb') as f:
    f.write(tflite_fp32)
fp32_mb = os.path.getsize(fp32_path) / (1024*1024)
print(f"  ✅ FP32 size: {fp32_mb:.3f} MB")

# ── INT8 export ───────────────────────────────────────────────────────────────
print("\nExporting INT8...")
X_train  = np.load(f"{DATA_DIR}X_train.npy").astype(np.float32)
rng      = np.random.default_rng(42)
rep_data = X_train[rng.choice(len(X_train), 200, replace=False)]

def representative_dataset():
    for s in rep_data:
        yield [s[np.newaxis]]

converter2 = tf.lite.TFLiteConverter.from_concrete_functions([concrete], model)
converter2.optimizations              = [tf.lite.Optimize.DEFAULT]
converter2.representative_dataset    = representative_dataset
converter2.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter2.inference_input_type      = tf.float32
converter2.inference_output_type     = tf.float32
tflite_int8 = converter2.convert()
int8_path   = f"{MODEL_DIR}model_int8.tflite"
with open(int8_path, 'wb') as f:
    f.write(tflite_int8)
int8_mb = os.path.getsize(int8_path) / (1024*1024)
print(f"  ✅ INT8 size: {int8_mb:.3f} MB")
print(f"  Size reduction: {fp32_mb/int8_mb:.1f}x smaller")

# ── Copy to android/assets ────────────────────────────────────────────────────
print("\nCopying to android/assets/...")
shutil.copy(fp32_path, f"{ANDROID_DIR}model_fp32.tflite")
shutil.copy(int8_path, f"{ANDROID_DIR}model_int8.tflite")
classes = np.load(f"{DATA_DIR}label_classes.npy", allow_pickle=True)
with open(f"{ANDROID_DIR}labels.txt", 'w') as f:
    for c in classes:
        f.write(c + '\n')
print("  Copied: model_fp32.tflite")
print("  Copied: model_int8.tflite")
print("  Copied: labels.txt")

with open(f"{RES_DIR}exp4_sizes.json", 'w') as f:
    json.dump({
        'fp32_mb':     round(fp32_mb, 4),
        'int8_mb':     round(int8_mb, 4),
        'reduction_x': round(fp32_mb/int8_mb, 2)
    }, f, indent=2)

print("\n✅ TFLite export COMPLETE!")
