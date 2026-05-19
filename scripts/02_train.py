import os

# using CPU because Metal/TensorFlow sometimes crashes on Mac
os.environ["TF_DEVICE_MIN_SYS_MEM_IN_MB"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report, confusion_matrix


DATA_DIR = "../data"
MODEL_DIR = "../models"
FIG_DIR = "../outputs/figures"
RESULTS_DIR = "../outputs/results"

EPOCHS = 50
BATCH_SIZE = 64
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

try:
    tf.config.set_visible_devices([], "GPU")
except:
    pass


print("Loading saved arrays...")

X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
classes = np.load(os.path.join(DATA_DIR, "label_classes.npy"), allow_pickle=True)

print("Train:", X_train.shape, y_train.shape)
print("Test:", X_test.shape, y_test.shape)
print("Classes:", list(classes))


with tf.device("/CPU:0"):
    model = models.Sequential([
        layers.Input(shape=(51, 3)),

        layers.Conv1D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),

        layers.Conv1D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),

        layers.Conv1D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),

        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.3),
        layers.Dense(len(classes), activation="softmax")
    ])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()
print("Parameters:", model.count_params())


train_callbacks = [
    callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=10,
        restore_best_weights=True
    ),
    callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, "model_best.keras"),
        monitor="val_accuracy",
        save_best_only=True
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
]


print("Training model...")

history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    callbacks=train_callbacks,
    verbose=1
)


loss, acc = model.evaluate(X_test, y_test, verbose=0)

print("Test accuracy:", round(acc * 100, 2))
print("Test loss:", round(loss, 4))


pred_probs = model.predict(X_test, verbose=0)

y_pred = np.argmax(pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

report = classification_report(
    y_true,
    y_pred,
    target_names=list(classes)
)

print(report)


cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=classes,
    yticklabels=classes
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "exp1_confusion_matrix.png"), dpi=150)
plt.close()


plt.figure(figsize=(7, 5))
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="validation")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "accuracy_curve.png"), dpi=150)
plt.close()


plt.figure(figsize=(7, 5))
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="validation")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "loss_curve.png"), dpi=150)
plt.close()


model.save(os.path.join(MODEL_DIR, "model.keras"))

results = {
    "test_accuracy": round(float(acc) * 100, 2),
    "test_loss": round(float(loss), 4),
    "total_params": int(model.count_params())
}

with open(os.path.join(RESULTS_DIR, "exp1_baseline.json"), "w") as f:
    json.dump(results, f, indent=2)

print("Done.")
