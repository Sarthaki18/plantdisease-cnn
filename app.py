
import io
import json
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import streamlit as st
from PIL import Image

try:
    import requests
except Exception:
    requests = None

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

st.set_page_config(
    page_title="LeafScan | Plant Disease AI",
    page_icon="LS",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "Classification-based Anomaly Detection"
MODEL_PATH = MODEL_DIR / "plant_disease_model_tf.h5"
KERAS_MODEL_PATH = MODEL_DIR / "plant_disease_model_tf.keras"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
DATASET_PATH = Path(r"D:\Plant_leave_diseases_dataset_with_augmentation")

# Insert Google Custom Search credentials here.
GOOGLE_API_KEY = "GOOGLE_API_KEY"
SEARCH_ENGINE_ID = "SEARCH_ENGINE_ID"

DEFAULT_CLASSES = sorted([
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Background_without_leaves", "Blueberry___healthy", "Cherry___healthy", "Cherry___Powdery_mildew",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot", "Corn___Common_rust", "Corn___healthy",
    "Corn___Northern_Leaf_Blight", "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
    "Grape___healthy", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot", "Peach___healthy", "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___healthy", "Potato___Late_blight", "Raspberry___healthy",
    "Soybean___healthy", "Squash___Powdery_mildew", "Strawberry___healthy", "Strawberry___Leaf_scorch",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___healthy", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_mosaic_virus", "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
])


def load_classes():
    if METADATA_PATH.exists():
        with METADATA_PATH.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        classes = metadata.get("classes") or []
        if classes:
            return classes, metadata
    if DATASET_PATH.exists():
        classes = sorted([p.name for p in DATASET_PATH.iterdir() if p.is_dir()])
        if classes:
            return classes, {"classes": classes, "source": str(DATASET_PATH)}
    return DEFAULT_CLASSES, {"classes": DEFAULT_CLASSES, "source": "fallback"}


CLASSES, METADATA = load_classes()
IMAGE_SIZE = tuple(METADATA.get("image_size", [224, 224])[:2]) if isinstance(METADATA.get("image_size"), list) else (224, 224)


@st.cache_resource(show_spinner=False)
def load_tf_model():
    if not TF_AVAILABLE:
        return None, "TensorFlow is not installed in this environment."

    model_file = MODEL_PATH if MODEL_PATH.exists() else KERAS_MODEL_PATH
    if not model_file.exists():
        return None, f"No trained TensorFlow model found at {MODEL_PATH}. Run python train_tf_model.py first."

    try:
        return tf.keras.models.load_model(model_file), None
    except Exception as exc:
        return None, f"Could not load TensorFlow model: {exc}"


def preprocess_image(image):
    image = image.convert("RGB").resize(IMAGE_SIZE)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def prettify_label(label):
    if "___" in label:
        plant, condition = label.split("___", 1)
    else:
        plant, condition = label, ""
    plant = plant.replace("_", " ").replace(",", ", ").strip()
    condition = condition.replace("_", " ").strip() or "Unknown"
    return plant, condition


def predict_disease(image):
    model, error = load_tf_model()
    if error:
        raise RuntimeError(error)

    probs = model.predict(preprocess_image(image), verbose=0)[0]
    top_indices = np.argsort(probs)[::-1][:5]
    return [(CLASSES[int(i)], float(probs[int(i)])) for i in top_indices]


def google_image_search(query_or_url):
    if requests is None:
        raise RuntimeError("The requests package is required for Google Custom Search calls.")
    if GOOGLE_API_KEY == "GOOGLE_API_KEY" or SEARCH_ENGINE_ID == "SEARCH_ENGINE_ID":
        raise RuntimeError("Insert GOOGLE_API_KEY and SEARCH_ENGINE_ID near the top of src/app.py.")

    params = urlencode({
        "key": GOOGLE_API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "searchType": "image",
        "q": query_or_url,
        "num": 5,
    })
    response = requests.get(f"https://www.googleapis.com/customsearch/v1?{params}", timeout=20)
    response.raise_for_status()
    return response.json().get("items", [])


st.markdown("""
<style>
:root { --bg:#f3f6ed; --ink:#152116; --muted:#60705f; --line:rgba(21,33,22,.14); --leaf:#2f7d32; --leaf-dark:#1d5e24; --cream:#fbfcf7; }
.stApp { background: var(--bg); color: var(--ink); }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding: 2rem 3rem 3rem; max-width: 1280px; }
h1, h2, h3 { color: var(--ink); letter-spacing: 0; }
.hero { min-height:310px; display:flex; align-items:flex-end; padding:32px; border-radius:8px; color:white; background:linear-gradient(90deg, rgba(10,23,12,.88), rgba(10,23,12,.38)), url('https://images.unsplash.com/photo-1464226184884-fa280b87c399?auto=format&fit=crop&w=1600&q=80'); background-size:cover; background-position:center; box-shadow:0 22px 60px rgba(21,33,22,.16); margin-bottom:24px; }
.hero h1 { color:white; font-size:clamp(2.4rem, 5vw, 5rem); line-height:.95; margin:0 0 10px; }
.hero p { color:#e9f4df; max-width:650px; font-size:1.05rem; }
.badge { display:inline-flex; padding:8px 12px; border-radius:999px; background:rgba(255,255,255,.16); color:white; font-weight:700; margin-bottom:14px; }
.panel { background:var(--cream); border:1px solid var(--line); border-radius:8px; padding:24px; box-shadow:0 12px 32px rgba(21,33,22,.08); }
.metric-card { background:#fff; border:1px solid var(--line); border-left:5px solid var(--leaf); border-radius:8px; padding:18px; }
.result-title { font-size:1.4rem; font-weight:800; margin:0; color:var(--leaf-dark); }
.result-subtitle { color:var(--muted); margin:4px 0 0; }
.stButton>button { background:var(--leaf)!important; color:white!important; border:0!important; border-radius:8px!important; min-height:46px; font-weight:800!important; }
.stButton>button:hover { background:var(--leaf-dark)!important; }
[data-testid="stFileUploader"] section { border:1.5px dashed rgba(47,125,50,.5); background:#fff; border-radius:8px; }
.section-label { color:var(--leaf); font-weight:900; text-transform:uppercase; font-size:.78rem; margin:0 0 6px; }
.footer { color:var(--muted); font-size:.9rem; padding-top:24px; }
@media (max-width:700px) { .block-container { padding:1rem; } .hero { min-height:240px; padding:22px; } }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero"><div><div class="badge">TensorFlow fine-tuned disease classification</div><h1>LeafScan</h1><p>Upload a plant leaf image, run the trained Keras model, and review disease predictions with confidence scores.</p></div></div>
""", unsafe_allow_html=True)

tab_detect, tab_search, tab_system = st.tabs(["Disease Detection", "Deceased Identification", "System"])

with tab_detect:
    left, right = st.columns([1.05, .95], gap="large")
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<p class="section-label">Image upload</p>', unsafe_allow_html=True)
        st.subheader("Analyze a leaf")
        uploaded_file = st.file_uploader("Upload JPG, JPEG, or PNG", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        analyze = st.button("Run disease prediction", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB") if uploaded_file else None
        if image:
            st.image(image, caption="Uploaded leaf image", use_container_width=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<p class="section-label">Model output</p>', unsafe_allow_html=True)
        st.subheader("Prediction results")
        if analyze:
            if image is None:
                st.warning("Upload an image before running prediction.")
            else:
                try:
                    with st.spinner("Running TensorFlow inference..."):
                        predictions = predict_disease(image)
                    top_label, top_conf = predictions[0]
                    plant, condition = prettify_label(top_label)
                    st.markdown(f'<div class="metric-card"><p class="result-title">{condition}</p><p class="result-subtitle">Plant: {plant} | Confidence: {top_conf * 100:.2f}%</p></div>', unsafe_allow_html=True)
                    st.write("")
                    st.write("Top predictions")
                    for label, conf in predictions:
                        plant_name, condition_name = prettify_label(label)
                        st.progress(conf, text=f"{plant_name} - {condition_name}: {conf * 100:.2f}%")
                except Exception as exc:
                    st.error(str(exc))
        else:
            st.info("Upload a leaf and run prediction to see live results here.")
        st.markdown('</div>', unsafe_allow_html=True)

with tab_search:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Google Custom Search API</p>', unsafe_allow_html=True)
    st.subheader("Deceased Identification")
    st.caption("Google Custom Search cannot reverse-search a private local file directly. Use a public image URL or query, then insert credentials in src/app.py.")
    public_url = st.text_input("Public image URL or identity search query")
    if st.button("Search identity", use_container_width=True):
        if not public_url.strip():
            st.warning("Enter a public image URL or query first.")
        else:
            try:
                items = google_image_search(public_url.strip())
                if not items:
                    st.info("No matching identity result found.")
                else:
                    best = items[0]
                    likely_name = (best.get("title") or "Unknown").split("|")[0].split("-")[0].strip()
                    st.success(f"Likely name: {likely_name}")
                    for item in items[:3]:
                        st.write(f"**{item.get('title', 'Untitled')}**")
                        st.caption(item.get("displayLink", "Unknown source"))
            except Exception as exc:
                st.error(str(exc))
    st.markdown('</div>', unsafe_allow_html=True)

with tab_system:
    model, load_error = load_tf_model()
    c1, c2, c3 = st.columns(3)
    c1.metric("Classes", len(CLASSES))
    c2.metric("Framework", "TensorFlow" if TF_AVAILABLE else "Unavailable")
    c3.metric("Input size", f"{IMAGE_SIZE[0]} x {IMAGE_SIZE[1]}")
    if load_error:
        st.warning(load_error)
    else:
        st.success(f"Model loaded from {MODEL_PATH if MODEL_PATH.exists() else KERAS_MODEL_PATH}")
    st.write("Dataset:", str(DATASET_PATH))
    st.write("Metadata source:", METADATA.get("dataset_path", METADATA.get("source", str(METADATA_PATH))))

st.markdown('<div class="footer">LeafScan | TensorFlow frontend relinked, backend app surface preserved.</div>', unsafe_allow_html=True)
