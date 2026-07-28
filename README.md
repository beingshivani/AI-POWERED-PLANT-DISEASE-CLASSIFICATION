# Plant Disease Classifier — Streamlit App

A simple Streamlit app that serves the CNN you trained in Colab (`best_model.keras`) for plant
leaf disease classification.

## 1. Get your model + labels out of Colab

Your `ModelCheckpoint` callback saved the best model to `/content/best_model.keras`, but that
file lives on Colab's temporary disk and disappears when the session ends. Before your session
resets, run this in a new cell at the bottom of your notebook:

```python
from google.colab import files
import json

# Save class labels (index -> class name), needed by the app
with open("labels.json", "w") as f:
    json.dump({v: k for k, v in train_generator.class_indices.items()}, f, indent=2)

# Download both files to your computer
files.download("best_model.keras")
files.download("labels.json")
```

(Alternative: mount Google Drive and `shutil.copy` the files there instead of downloading,
if you'd rather grab them later.)

## 2. Set up this project folder

Place the two downloaded files directly inside this folder, next to `app.py`, so it looks like:

```
plant-disease-app/
├── app.py
├── requirements.txt
├── README.md
├── best_model.keras   <- your downloaded model
└── labels.json         <- your downloaded labels
```

## 3. Run it locally (optional but recommended)

```bash
cd plant-disease-app
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

It'll open at `http://localhost:8501`.

## 4. Push to GitHub

> ⚠️ **Model file size**: If `best_model.keras` is over 100 MB, GitHub will reject a normal
> push. Check its size with `ls -lh best_model.keras`. If it's too big, use
> [Git LFS](https://git-lfs.com/) (`git lfs install && git lfs track "*.keras"`) before
> committing, or host the model file elsewhere (e.g. Google Drive/Hugging Face Hub) and
> download it at app startup instead.

```bash
cd plant-disease-app
git init
git add .
git commit -m "Plant disease classifier Streamlit app"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

(Create the empty repo on GitHub first via github.com/new, without a README, so there's no
merge conflict.)

## 5. Deploy on Streamlit Community Cloud (free)

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **New app**.
3. Pick your repo, branch (`main`), and set the main file path to `app.py`.
4. Click **Deploy**.

Your app will build (installing `requirements.txt`) and go live at a public
`https://<something>.streamlit.app` URL within a couple of minutes.

## Notes

- `IMG_SIZE` in `app.py` is set to 128 to match your training pipeline — don't change it unless
  you retrain at a different size.
- The model is loaded with `@st.cache_resource` so it only loads once per app instance, not on
  every user interaction.
- If TensorFlow install is slow/large on deploy, `tensorflow-cpu` (already in
  `requirements.txt`) keeps things lighter than the full `tensorflow` GPU package, which isn't
  needed for inference anyway.
