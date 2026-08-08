# Plant Disease Classifier — Streamlit App

A Streamlit app that serves a CNN trained for plant leaf disease classification,
plus a training script (`train.py`) with data augmentation, and an in-app
evaluation tab for measuring model performance on a labeled test set.

## Project structure

```
plant-disease-app/
├── app.py                 <- Streamlit app: prediction + evaluation UI
├── train.py                <- Training script (run in Colab or locally with a GPU)
├── requirements.txt
├── README.md
├── model.weights.h5        <- trained weights (weights-only, not full model)
└── labels.json             <- class index -> class name mapping
```

**Note:** this project saves and loads **weights only** (`model.weights.h5`), not a
full model file (e.g. `.keras` or `.h5` full-model format). This means `app.py`
must rebuild the *exact* training-time architecture (see `build_model()` in both
`app.py` and `train.py`, which are kept identical) before calling
`model.load_weights(...)`. If you ever change the architecture in one file, you
must change it identically in the other, or weight loading will fail or silently
load incorrectly.

---

## 1. Train the model (produces `model.weights.h5` + `labels.json`)

Run `train.py`, either locally or in Google Colab (recommended, for free GPU access).

### Expected dataset structure

Organize your dataset into `train/` and `val/` folders, each containing one
subfolder per class:

```
data/
├── train/
│   ├── Apple_healthy/
│   │   ├── img1.jpg
│   │   └── ...
│   ├── Apple_scab/
│   └── ...
└── val/
    ├── Apple_healthy/
    ├── Apple_scab/
    └── ...
```

### Run training

```
python train.py --train_dir data/train --val_dir data/val --epochs 25
```

This will:
- Apply data augmentation (rotation, shift, shear, zoom, horizontal flip,
  brightness jitter) to the training set only — the validation set stays
  unaugmented, since it needs to reflect real, unaltered images to give an
  honest read on generalization.
- Train the CNN, using early stopping on validation loss to avoid overfitting.
- Save the best-performing weights to `model.weights.h5` (weights-only,
  matching what `app.py` expects).
- Save the class index → name mapping to `labels.json`.

If training in Colab, download both output files before your session ends:

```python
from google.colab import files
files.download("model.weights.h5")
files.download("labels.json")
```

---

## 2. Set up this project folder

Place `model.weights.h5` and `labels.json` directly inside this folder, next to
`app.py`, matching the structure shown above.

---

## 3. Run the app locally

```
cd plant-disease-app
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

It'll open at `http://localhost:8501`, with two tabs:

- **Predict** — upload a single leaf photo, get back the predicted class and
  confidence, plus a full probability breakdown across all classes.
- **Evaluate on Test Set** — upload a `.zip` of a labeled test set (same
  folder-per-class structure as training data) to compute cross-entropy loss,
  accuracy, precision, recall, F1 score (weighted), a per-class classification
  report, and a confusion matrix.

---

## 4. Push to GitHub

> ⚠️ **Model file size**: if `model.weights.h5` is over 100 MB, GitHub will
> reject a normal push. Check its size with `ls -lh model.weights.h5`. If it's
> too big, use [Git LFS](https://git-lfs.com/)
> (`git lfs install && git lfs track "*.h5"`) before committing, or host the
> file elsewhere (e.g. Hugging Face Hub) and download it at app startup instead.

```
cd plant-disease-app
git init
git add .
git commit -m "Plant disease classifier: app, training script, evaluation"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

---

## 5. Deploy on Streamlit Community Cloud (free)

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. Click **New app**.
3. Pick your repo, branch (`main`), and set the main file path to `app.py`.
4. Click **Deploy**.

Your app will build (installing `requirements.txt`) and go live at a public
`https://<something>.streamlit.app` URL within a couple of minutes.

---

## Notes

- `IMG_SIZE` is set to 128 in both `app.py` and `train.py` — keep these in sync;
  don't change one without the other and retraining.
- Both the model and labels are loaded with `@st.cache_resource` in `app.py`,
  so they only load once per app session, not on every user interaction.
- If TensorFlow install is slow/large on deploy, `tensorflow-cpu` (in
  `requirements.txt`) keeps things lighter than the full `tensorflow` GPU
  package, which isn't needed for inference anyway.
- No data augmentation is applied at inference time in `app.py` — augmentation
  only ever applies during training (`train.py`), to the training split only.
- Known limitation: this model was trained from scratch, not via transfer
  learning from a pretrained backbone (e.g. MobileNetV2, EfficientNet) — a
  likely accuracy improvement if revisited.
