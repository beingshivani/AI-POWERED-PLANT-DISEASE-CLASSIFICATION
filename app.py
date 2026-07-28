import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
)

# ---------- Config ----------
WEIGHTS_PATH = "model.weights.h5"   # weights-only file downloaded from Colab
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


# ---------- UI ----------
st.title("🌿 Plant Disease Classifier")
st.write("Upload a photo of a plant leaf and the model will predict the disease (or healthy) class.")

labels = load_labels()
model = load_model(len(labels))

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
