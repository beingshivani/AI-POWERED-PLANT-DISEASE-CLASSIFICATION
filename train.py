"""
train.py

Trains the plant disease classification CNN with data augmentation applied
to the training set. This is a TRAINING script, meant to run once (e.g. in
Colab, with GPU access) to produce model.weights.h5 -- the file app.py loads
at inference time.

The architecture here must stay IDENTICAL to build_model() in app.py, since
app.py rebuilds this exact structure and loads these weights into it.

Usage:
    python train.py --train_dir data/train --val_dir data/val --epochs 25
"""

import argparse
import json

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

IMG_SIZE = 128  # must match IMG_SIZE in app.py exactly
BATCH_SIZE = 32


def build_model(num_classes: int) -> tf.keras.Model:
    """
    Identical to build_model() in app.py. Keeping this in sync is essential --
    app.py rebuilds this architecture from scratch and loads trained weights
    into it, so any mismatch here will break weight loading in the app.
    """
    model = Sequential([
        Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

        Conv2D(32, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Conv2D(256, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ])
    return model


def build_data_generators(train_dir: str, val_dir: str):
    """
    Builds the training and validation data generators.

    Augmentation is applied ONLY to the training generator -- the validation
    set must reflect real, unaltered images, since its job is to measure how
    well the model generalizes, not to be augmented itself. Applying
    augmentation to validation data would give a misleadingly optimistic or
    noisy read on model performance.
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,          # normalize pixel values to 0-1, same as app.py's preprocess_image
        rotation_range=25,          # randomly rotate images up to 25 degrees
        width_shift_range=0.15,     # randomly shift image horizontally by up to 15% of width
        height_shift_range=0.15,    # randomly shift image vertically by up to 15% of height
        shear_range=0.1,            # randomly apply shear (slanting) transformations
        zoom_range=0.2,             # randomly zoom in/out by up to 20%
        horizontal_flip=True,       # randomly flip images horizontally
        brightness_range=[0.8, 1.2],  # randomly adjust brightness between 80%-120% of original
        fill_mode="nearest",        # how to fill in newly created pixels from rotation/shift
    )

    # Validation data: rescale ONLY, no augmentation -- this must stay a clean,
    # unaltered reflection of real-world images to give an honest performance signal.
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="sparse",   # integer labels, matches SparseCategoricalCrossentropy
        shuffle=True,
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="sparse",
        shuffle=False,  # no need to shuffle validation data
    )

    return train_generator, val_generator


def train(train_dir: str, val_dir: str, epochs: int, output_weights: str, labels_path: str):
    train_generator, val_generator = build_data_generators(train_dir, val_dir)
    num_classes = len(train_generator.class_indices)

    print(f"Found {num_classes} classes: {train_generator.class_indices}")

    # Save labels.json in the same format app.py expects: {index: class_name}
    idx_to_class = {v: k for k, v in train_generator.class_indices.items()}
    with open(labels_path, "w") as f:
        json.dump(idx_to_class, f, indent=2)
    print(f"Saved label mapping to {labels_path}")

    model = build_model(num_classes)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",  # matches app.py's evaluation loss
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        # Stop training early if validation loss stops improving, to avoid overfitting
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        # Save the best-performing weights seen during training, not just the final epoch
        ModelCheckpoint(output_weights, monitor="val_loss", save_best_only=True, save_weights_only=True),
    ]

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
    )

    print(f"Training complete. Best weights saved to {output_weights}")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the plant disease classifier with data augmentation.")
    parser.add_argument("--train_dir", type=str, required=True,
                         help="Path to training data (folder-per-class structure).")
    parser.add_argument("--val_dir", type=str, required=True,
                         help="Path to validation data (folder-per-class structure).")
    parser.add_argument("--epochs", type=int, default=25,
                         help="Maximum number of training epochs.")
    parser.add_argument("--output_weights", type=str, default="model.weights.h5",
                         help="Where to save the trained weights file.")
    parser.add_argument("--labels_path", type=str, default="labels.json",
                         help="Where to save the class index -> name mapping.")
    args = parser.parse_args()

    train(args.train_dir, args.val_dir, args.epochs, args.output_weights, args.labels_path)
