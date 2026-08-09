# -*- coding: utf-8 -*-
"""train.py
 
Trains the plant disease classification CNN and saves weights-only, matching
the format app.py expects (model.load_weights(), not tf.keras.models.load_model()).
 
Uses class_mode='sparse' + sparse_categorical_crossentropy throughout, so the
label format here matches app.py's evaluation code (which builds y_true as
plain integers, not one-hot vectors).
 
Original file is located at
    https://colab.research.google.com/drive/1aN_YhLyGV1qN2cGVX53DIb31lsPNvDLF
"""
 
from google.colab import files
uploaded = files.upload()
 
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!pip install -q kaggle
 
!kaggle datasets download -d saroz014/plant-disease -p /content/raw --unzip
 
import os, shutil
 
dup_path = "/content/raw/dataset/dataset"
if os.path.exists(dup_path):
    shutil.rmtree(dup_path)
    print("Removed duplicate nested folder")
 
fixed = 0
for dirpath, dirnames, filenames in os.walk("/content/raw/dataset"):
    for fname in filenames:
        name, ext = os.path.splitext(fname)
        stripped = name.strip()
        if stripped != name:
            os.rename(os.path.join(dirpath, fname), os.path.join(dirpath, stripped + ext))
            fixed += 1
print(f"Fixed {fixed} filenames")
 
print(os.listdir("/content/raw/dataset"))
print(os.listdir("/content/raw/dataset/train"))
 
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
 
print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))
 
train_dir = "/content/raw/dataset/train"
test_dir = "/content/raw/dataset/test"
 
IMG_SIZE = 128  # must match IMG_SIZE in app.py exactly
BATCH_SIZE = 32
 
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)
 
test_datagen = ImageDataGenerator(rescale=1./255)
 
# class_mode='sparse' -> labels come out as plain integers (0, 1, 2, ...),
# matching the format app.py's evaluation code builds y_true in, and pairing
# with sparse_categorical_crossentropy below instead of the one-hot
# 'categorical' format used previously.
train_generator, val_generator = (
    train_datagen.flow_from_directory(train_dir, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='sparse', subset=subset)
    for subset in ['training', 'validation']
)
 
test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='sparse', shuffle=False
)
 
num_classes = train_generator.num_classes
print("Number of classes:", num_classes)
print("Class indices:", train_generator.class_indices)
 
model = Sequential([
    Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    Conv2D(32, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
 
    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
])
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
 
model.add(Conv2D(256, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
 
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
 
model.summary()
 
# Recompile with a lower learning rate for more careful/stable training steps.
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
 
callbacks = [
    # Stop training early if validation loss stops improving, to avoid overfitting
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    # save_weights_only=True -> writes only the weight values (matching what
    # app.py expects via model.load_weights()), not the full model.
    ModelCheckpoint('model.weights.h5', monitor='val_accuracy', save_best_only=True, save_weights_only=True)
]
 
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=300,
    callbacks=callbacks
)
 
# Download the trained weights -- Colab's disk is temporary and this file
# disappears when the session ends.
files.download("model.weights.h5")
 
import json
with open('labels.json', 'w') as f:
    json.dump({v: k for k, v in train_generator.class_indices.items()}, f, indent=2)
files.download('labels.json')
 
# ---------- Evaluate on the held-out test set ----------
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix, classification_report
)
 
y_true = test_generator.classes          # true integer labels, in generator order
probs = model.predict(test_generator, verbose=1)
y_pred = np.argmax(probs, axis=1)        # convert predicted probabilities to class indices
 
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
 
print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
 
idx_to_class = {v: k for k, v in test_generator.class_indices.items()}
target_names = [idx_to_class[i] for i in range(len(idx_to_class))]
 
print("\nPer-Class Report:")
print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
 
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)
