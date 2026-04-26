import os
os.environ['TF_DEVICE_MIN_SYS_MEM_IN_MB'] = '0'
# Force CPU to avoid Metal GPU crash on M3
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress info logs

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Force CPU
tf.config.set_visible_devices([], 'GPU')

DATA_DIR   = "../data/"
MODEL_DIR  = "../models/"
FIG_DIR    = "../outputs/figures/"
RES_DIR    = "../outputs/results/"
EPOCHS     = 50
BATCH_SIZE = 64
SEED       = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR,   exist_ok=True)
os.makedirs(RES_DIR,   exist_ok=True)

print("Loading data...")
X_train = np.load(f"{DATA_DIR}X_train.npy")
X_test  = np.load(f"{DATA_DIR}X_test.npy")
y_train = np.load(f"{DATA_DIR}y_train.npy")
y_test  = np.load(f"{DATA_DIR}y_test.npy")
classes = np.load(f"{DATA_DIR}label_classes.npy", allow_pickle=True)
print(f"  X_train: {X_train.shape}")
print(f"  X_test:  {X_test.shape}")
print(f"  Classes: {list(classes)}")

print("\nBuilding 1D-CNN model...")
with tf.device('/CPU:0'):
    model = models.Sequential([
        layers.Input(shape=(51, 3)),
        layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.3),
        layers.Dense(6, activation='softmax', name='classifier')
    ], name='HAR_1DCNN')

model.summary()
print(f"\nTotal parameters: {model.count_params():,}")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

cb_list = [
    callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ModelCheckpoint(
        filepath=f"{MODEL_DIR}model_best.keras",
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]

print("\nTraining on CPU... (takes 5-15 minutes)")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    callbacks=cb_list,
    verbose=1
)

print("\nEvaluating on test set...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {accuracy*100:.2f}%")
print(f"   Test Loss:     {loss:.4f}")

y_pred     = model.predict(X_test, verbose=0)
y_pred_cls = np.argmax(y_pred, axis=1)
y_true_cls = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_true_cls, y_pred_cls, target_names=list(classes)))

cm = confusion_matrix(y_true_cls, y_pred_cls)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix — E1 WISDM Baseline')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(f"{FIG_DIR}exp1_confusion_matrix.png", dpi=150)
print("Saved confusion matrix.")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history.history['accuracy'],     label='Train')
ax1.plot(history.history['val_accuracy'], label='Val')
ax1.set_title('Accuracy')
ax1.legend()
ax2.plot(history.history['loss'],     label='Train')
ax2.plot(history.history['val_loss'], label='Val')
ax2.set_title('Loss')
ax2.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}training_curves.png", dpi=150)
print("Saved training curves.")

model.save(f"{MODEL_DIR}model.keras")
print("Model saved.")

with open(f"{RES_DIR}exp1_baseline.json", 'w') as f:
    json.dump({
        'test_accuracy': round(float(accuracy)*100, 2),
        'test_loss':     round(float(loss), 4),
        'total_params':  int(model.count_params())
    }, f, indent=2)

print("\n✅ Training COMPLETE!")
