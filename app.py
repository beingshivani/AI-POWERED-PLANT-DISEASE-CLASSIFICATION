"""
app.py

Streamlit web app for the plant disease classifier.

Loads the CNN trained by train.py (model.weights.h5 + labels.json) and provides:
  - Predict tab: upload a single leaf photo, get a disease prediction.
  - Evaluate tab: upload a zip of labeled test images, get accuracy/precision/
    recall/F1, a per-class report, and a confusion matrix.

The build_model() architecture here MUST stay identical to build_model() in
train.py -- this app rebuilds that exact structure and loads the trained
weights into it. Any mismatch will break weight loading.

Run with:
    streamlit run app.py
"""

import json
import os
import shutil
import tempfile
import zipfile

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
)

# ---------- Config ----------
# NOTE: this filename must END in ".weights.h5" -- Keras picks its file-reading
# logic based on that exact suffix. Naming it "model_weights.h5" (missing the
# ".weights." part) will fail to load even though the file itself is fine.
WEIGHTS_PATH = "model.weights.h5"
LABELS_PATH = "labels.json"
IMG_SIZE = 128  # must match IMG_SIZE used during training

st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌿", layout="centered")


# ---------- Load labels first (needed to know num_classes for the model) ----------
@st.cache_resource
def load_labels():
    with open(LABELS_PATH, "r") as f:
        raw = json.load(f)
    # json keys are always strings, convert back to int -> class name
    return {int(k): v for k, v in raw.items()}


# ---------- Rebuild the exact training architecture, then load weights into it ----------
def build_model(num_classes: int) -> tf.keras.Model:
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


@st.cache_resource
def load_model(num_classes: int):
    model = build_model(num_classes)
    model.load_weights(WEIGHTS_PATH)
    return model


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


# ---------- Evaluation helpers ----------
def extract_zip_to_tempdir(uploaded_zip) -> str:
    """
    Extracts an uploaded zip file to a temporary directory and returns its path.
    Expected structure inside the zip:
        class_name_1/
            img1.jpg
            img2.jpg
        class_name_2/
            img3.jpg
            ...
    This mirrors the same folder-per-class structure used by
    ImageDataGenerator.flow_from_directory during training.
    """
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(uploaded_zip) as zf:
        zf.extractall(temp_dir)
    return temp_dir


def find_class_root(temp_dir: str) -> str:
    """
    Zip files often extract into a single wrapper folder (e.g. 'test_data/').
    This walks down one level if the top of temp_dir contains only one folder
    and no images directly, so class subfolders are found correctly either way.
    """
    entries = [e for e in os.listdir(temp_dir) if not e.startswith("__MACOSX")]
    full_paths = [os.path.join(temp_dir, e) for e in entries]
    only_one_folder = len(full_paths) == 1 and os.path.isdir(full_paths[0])
    if only_one_folder:
        return full_paths[0]
    return temp_dir


def run_evaluation(model, labels: dict, class_root: str):
    """
    Runs the model over every image in every class subfolder under class_root,
    collecting true labels, predicted labels, and predicted probabilities.

    Returns:
        y_true: list[int] - ground-truth class indices (from folder names)
        y_pred: list[int] - predicted class indices
        y_probs: list[np.ndarray] - full softmax probability vector per image
        skipped: list[str] - folder names that didn't match any known class
    """
    name_to_idx = {v: k for k, v in labels.items()}

    y_true, y_pred, y_probs = [], [], []
    skipped = []

    class_folders = sorted([
        d for d in os.listdir(class_root)
        if os.path.isdir(os.path.join(class_root, d))
    ])

    progress_bar = st.progress(0, text="Starting evaluation...")
    total_folders = len(class_folders)

    for folder_i, class_name in enumerate(class_folders):
        if class_name not in name_to_idx:
            skipped.append(class_name)
            continue

        true_idx = name_to_idx[class_name]
        folder_path = os.path.join(class_root, class_name)
        image_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            try:
                image = Image.open(img_path)
                x = preprocess_image(image)
                probs = model.predict(x, verbose=0)[0]
            except Exception:
                continue  # skip unreadable/corrupt files rather than crash the run

            pred_idx = int(np.argmax(probs))

            y_true.append(true_idx)
            y_pred.append(pred_idx)
            y_probs.append(probs)

        progress_bar.progress(
            (folder_i + 1) / total_folders,
            text=f"Evaluated class: {class_name} ({folder_i + 1}/{total_folders})"
        )

    progress_bar.empty()
    return y_true, y_pred, y_probs, skipped


def compute_metrics(y_true, y_pred, y_probs, num_classes: int):
    """
    Computes cross-entropy loss, precision, recall, and F1 (weighted average,
    which accounts for class imbalance by weighting each class's score by its
    support -- how many true examples of that class exist in the test set).
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    y_probs_arr = np.array(y_probs)

    cce = SparseCategoricalCrossentropy()
    loss_value = float(cce(y_true_arr, y_probs_arr).numpy())

    precision = precision_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)
    recall = recall_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)
    f1 = f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)
    accuracy = float(np.mean(y_true_arr == y_pred_arr))

    return {
        "loss": loss_value,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


# ---------- UI ----------
st.title("🌿 Plant Disease Classifier")

# Fail fast with a clear message if the model files aren't next to app.py --
# this is the most common setup mistake, so surface it clearly instead of a
# confusing stack trace.
missing = [p for p in (WEIGHTS_PATH, LABELS_PATH) if not os.path.exists(p)]
if missing:
    st.error(
        f"Missing required file(s): {', '.join(missing)}. "
        f"Place model.weights.h5 and labels.json in the same folder as app.py "
        f"(produced by running train.py)."
    )
    st.stop()

labels = load_labels()
model = load_model(len(labels))

predict_tab, evaluate_tab = st.tabs(["Predict", "Evaluate on Test Set"])

# ===================== PREDICT TAB =====================
with predict_tab:
    st.write("Upload a photo of a plant leaf and the model will predict the disease (or healthy) class.")

    uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_container_width=True)

        with st.spinner("Analyzing..."):
            x = preprocess_image(image)
            preds = model.predict(x)[0]
            top_idx = int(np.argmax(preds))
            confidence = float(preds[top_idx]) * 100

        st.subheader("Prediction")
        st.success(f"**{labels[top_idx]}** ({confidence:.2f}% confidence)")

        with st.expander("See full class probabilities"):
            sorted_idx = np.argsort(preds)[::-1]
            for i in sorted_idx:
                st.write(f"{labels[i]}: {preds[i] * 100:.2f}%")
    else:
        st.info("Upload an image to get a prediction.")

# ===================== EVALUATE TAB =====================
with evaluate_tab:
    st.write(
        "Upload a **zip file** containing a labeled test set to evaluate the model's "
        "performance. The zip should contain one folder per class, matching your "
        "training class names, with test images inside each folder — the same "
        "structure `flow_from_directory` expects."
    )
    st.caption(
        "Example structure:\n\n"
        "test_data.zip\n"
        "├── Apple___healthy/\n"
        "│   ├── img1.jpg\n"
        "├── Apple___Apple_scab/\n"
        "│   ├── img2.jpg\n"
        "└── ..."
    )

    uploaded_zip = st.file_uploader("Upload test set (.zip)", type=["zip"], key="eval_zip")

    if uploaded_zip is not None:
        if st.button("Run Evaluation"):
            with st.spinner("Extracting zip file..."):
                temp_dir = extract_zip_to_tempdir(uploaded_zip)
                class_root = find_class_root(temp_dir)

            y_true, y_pred, y_probs, skipped = run_evaluation(model, labels, class_root)

            shutil.rmtree(temp_dir, ignore_errors=True)

            if skipped:
                st.warning(
                    f"Skipped {len(skipped)} folder(s) that didn't match any known "
                    f"class name: {', '.join(skipped)}"
                )

            if len(y_true) == 0:
                st.error(
                    "No evaluable images found. Check that folder names inside the "
                    "zip exactly match your training class names."
                )
            else:
                metrics = compute_metrics(y_true, y_pred, y_probs, len(labels))

                st.subheader("Overall Metrics")
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
                col2.metric("Precision", f"{metrics['precision']:.4f}")
                col3.metric("Recall", f"{metrics['recall']:.4f}")
                col4.metric("F1 Score", f"{metrics['f1']:.4f}")
                col5.metric("Cross-Entropy Loss", f"{metrics['loss']:.4f}")

                st.caption(
                    "Precision, Recall, and F1 use weighted averaging across classes "
                    "(each class's score weighted by how many true examples of it "
                    "exist in the test set), which accounts for class imbalance."
                )

                st.subheader("Per-Class Report")
                target_names = [labels[i] for i in sorted(labels.keys())]
                report_dict = classification_report(
                    y_true, y_pred,
                    labels=sorted(labels.keys()),
                    target_names=target_names,
                    output_dict=True,
                    zero_division=0,
                )
                report_df = pd.DataFrame(report_dict).transpose()
                st.dataframe(report_df.style.format("{:.3f}", subset=report_df.columns[:-1]))

                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_true, y_pred, labels=sorted(labels.keys()))
                cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
                st.dataframe(cm_df)

                st.caption(
                    f"Evaluated on {len(y_true)} images across "
                    f"{len(set(y_true))} classes."
                )
    else:
        st.info("Upload a zip file of labeled test images to run evaluation.")
