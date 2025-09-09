# app.py
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import time
import os
import pandas as pd
import gdown

# -----------------------
# CONFIGURATION
# -----------------------
APP_TITLE = "Alzheimer's Disease Classifier"
LOGO_URL = "https://img.icons8.com/color/96/brain.png"
UPLOAD_ICON_URL = "https://img.icons8.com/fluency/48/upload.png"
BG_URL = "https://images.unsplash.com/photo-1617791160505-6f00504e3519?w=1600&q=80&auto=format&fit=crop"  

MODEL_PATH = "best_model.h5"
FILE_ID = "1JLB1DxdwMWgAKM2dpoF0V-Heja-ePbGO"
DRIVE_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

MODEL_INPUT_SIZE = (150, 150)
CLASS_NAMES = [
    "MildDemented",
    "ModerateDemented",
    "NonDemented",
    "VeryMildDemented"
]
# -----------------------

st.set_page_config(
    page_title="Alzheimer's",
    layout="centered"
)

# CSS for background + center box
page_bg = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background: url("{BG_URL}");
    background-size: cover;
    background-position: center;
}}
.block-container {{
    background: rgba(255, 255, 255, 0.8);
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.2);
    max-width: 800px;
    margin: 2rem auto;
}}
/* Hide Streamlit default header & menu */
header[data-testid="stHeader"] {{
    display: none;
}}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)


# Header dengan logo
col1, col2 = st.columns([1, 5]) 

with col1:
    st.image(LOGO_URL, width=80)

with col2:
    st.markdown(
        f"<h1 style='margin-top: 0px;'>{APP_TITLE}</h1>",
        unsafe_allow_html=True
    )


# Deskripsi Alzheimer
st.markdown("""
**Penyakit Alzheimer** adalah kondisi otak degeneratif yang menyebabkan penurunan progresif dalam sejumlah aspek.
Mulai dari ingatan, kognitif atau kemampuan berpikir, kemampuan bicara dan perilaku.
Penyakit ini dapat menyasar orang dewasa yang masih muda. Namun, sebagian besar kasusnya terjadi pada 
mereka yang berusia lebih dari 60 tahun (lansia) (halodoc.com, 2023)
""")

# Load model
@st.cache_resource(show_spinner=True)
def load_model():
    try:
        # Download model kalau belum ada
        if not os.path.exists(MODEL_PATH):
            with st.spinner("Downloading model from Google Drive..."):
                gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)
        
        # Load model
        model = tf.keras.models.load_model(MODEL_PATH)
        return model, None
    except Exception as e:
        return None, f"Error loading model: {e}"

model, model_error = load_model()

# Preprocessing
def preprocess_image(pil_img, target_size):
    img = ImageOps.exif_transpose(pil_img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size, Image.BILINEAR)
    arr = np.asarray(img).astype("float32") / 255.0
    return arr

# Upload section
st.subheader("Upload Brain MRI Image")
uploaded = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded image", width=150)
    if st.button("🔍 Predict"):
        with st.spinner("Predicting..."):
            try:
                x = preprocess_image(img, MODEL_INPUT_SIZE)
                x_batch = np.expand_dims(x, axis=0)
                preds = model.predict(x_batch)

                if preds.ndim == 2 and preds.shape[1] >= 1:
                    probs = tf.nn.softmax(preds[0]).numpy()
                elif preds.ndim == 1:
                    probs = tf.nn.softmax(preds).numpy()
                else:
                    probs = preds[0]

                top_idx = int(np.argmax(probs))
                top_label = CLASS_NAMES[top_idx] if top_idx < len(CLASS_NAMES) else f"Class {top_idx}"

                st.success(f"Prediction: **{top_label}**")
                st.write(f"Confidence: **{probs[top_idx]*100:.2f}%**")

                df = pd.DataFrame({
                    "class": [CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class {i}" for i in range(len(probs))],
                    "probability": probs
                }).sort_values("probability", ascending=False)
                st.table(df.style.format({"probability": "{:.4f}"}))

            except Exception as e:
                st.error(f"Error saat prediksi: {e}")
else:
    st.info("Silakan unggah gambar untuk diprediksi.")
