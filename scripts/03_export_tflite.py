import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
import shutil
import numpy as np
import tensorflow as tf


DATA_DIR = "../data"
MODEL_DIR = "../models"
ANDROID_DIR = "../android/assets"
RESULTS_DIR = "../outputs/results"

os.makedirs(ANDROID_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

try:
    tf.config.set_visible_devices([], "GPU")
except:
    pass


model_path = os.path.join(MODEL_DIR, "model.keras")

print("Loading model...")
model = tf.keras.models.load_model(model_path)


# convert main model
input_spec = tf.TensorSpec(shape=[1, 51, 3], dtype=tf.float32)
model_func = tf.function(lambda x: model(x))
concrete_func = model_func.get_concrete_function(input_spec)

converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [concrete_func],
    model
)

tflite_model = converter.convert()

fp32_path = os.path.join(MODEL_DIR, "model_fp32.tflite")

with open(fp32_path, "wb") as f:
    f.write(tflite_model)

fp32_size = os.path.getsize(fp32_path) / (1024 * 1024)
print("FP32 model size:", round(fp32_size, 4), "MB")


# quantized model
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy")).astype("float32")

rng = np.random.default_rng(42)
sample_count = min(200, len(X_train))
sample_idx = rng.choice(len(X_train), sample_count, replace=False)
calibration_data = X_train[sample_idx]


def representative_data():
    for sample in calibration_data:
        yield [sample[np.newaxis, :, :]]


converter_q = tf.lite.TFLiteConverter.from_concrete_functions(
    [concrete_func],
    model
)

converter_q.optimizations = [tf.lite.Optimize.DEFAULT]
converter_q.representative_dataset = representative_data
converter_q.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# keep input/output as float so Android side is easier
converter_q.inference_input_type = tf.float32
converter_q.inference_output_type = tf.float32

tflite_quant = converter_q.convert()

int8_path = os.path.join(MODEL_DIR, "model_int8.tflite")

with open(int8_path, "wb") as f:
    f.write(tflite_quant)

int8_size = os.path.getsize(int8_path) / (1024 * 1024)

print("INT8 model size:", round(int8_size, 4), "MB")

if int8_size > 0:
    print("Reduction:", round(fp32_size / int8_size, 2), "x")


# copy files for android app
shutil.copy(fp32_path, os.path.join(ANDROID_DIR, "model_fp32.tflite"))
shutil.copy(int8_path, os.path.join(ANDROID_DIR, "model_int8.tflite"))

classes = np.load(
    os.path.join(DATA_DIR, "label_classes.npy"),
    allow_pickle=True
)

label_path = os.path.join(ANDROID_DIR, "labels.txt")

with open(label_path, "w") as f:
    for name in classes:
        f.write(str(name) + "\n")


results = {
    "fp32_mb": round(float(fp32_size), 4),
    "int8_mb": round(float(int8_size), 4),
    "reduction_x": round(float(fp32_size / int8_size), 2)
}

with open(os.path.join(RESULTS_DIR, "exp4_sizes.json"), "w") as f:
    json.dump(results, f, indent=2)

print("Files copied to android assets.")
print("TFLite export finished.")
