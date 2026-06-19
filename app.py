import io
import importlib.util
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

TF_AVAILABLE = importlib.util.find_spec("tensorflow") is not None

st.set_page_config(
    page_title="LeafScan | Plant Disease AI",
    page_icon="LS",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE = Path(__file__).resolve().parent
PROJECT_ROOT = BASE.parent
MODEL_DIR = BASE / "Classification-based Anomaly Detection"
MODEL_PATH = PROJECT_ROOT / "plant_disease_model.keras"
FALLBACK_H5_MODEL_PATH = MODEL_DIR / "plant_disease_model_tf.h5"
FALLBACK_KERAS_MODEL_PATH = MODEL_DIR / "plant_disease_model_tf.keras"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
DATASET_PATH = Path(r"D:\Plant_leave_diseases_dataset_with_augmentation")

GOOGLE_API_KEY = "GOOGLE_API_KEY"
SEARCH_ENGINE_ID = "SEARCH_ENGINE_ID"

DEFAULT_CLASSES = sorted([
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Background_without_leaves", "Blueberry___healthy", "Cherry_(including_sour)___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
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
IMAGE_SIZE = tuple(METADATA.get("image_size", (224, 224)))
BACKGROUND_CLASS = "Background_without_leaves"
PLANT_ALIASES = {
    "apple": "Apple",
    "blueberry": "Blueberry",
    "cherry": "Cherry",
    "corn": "Corn",
    "maize": "Corn",
    "grape": "Grape",
    "orange": "Orange",
    "peach": "Peach",
    "pepper": "Pepper, bell",
    "bell pepper": "Pepper, bell",
    "potato": "Potato",
    "raspberry": "Raspberry",
    "soybean": "Soybean",
    "squash": "Squash",
    "strawberry": "Strawberry",
    "tomato": "Tomato",
    "septoria": "Tomato",
    "leaf mold": "Tomato",
    "yellow leaf curl": "Tomato",
    "mosaic": "Tomato",
}


@st.cache_resource(show_spinner=False)
def load_tf_model():
    if not TF_AVAILABLE:
        return None, "TensorFlow is not installed in this environment."

    try:
        import tensorflow as tf
    except Exception as exc:
        return None, f"TensorFlow could not be imported: {exc}"

    model_candidates = [MODEL_PATH, FALLBACK_KERAS_MODEL_PATH, FALLBACK_H5_MODEL_PATH]
    model_file = next((path for path in model_candidates if path.exists()), None)
    if model_file is None:
        searched = ", ".join(str(path) for path in model_candidates)
        return None, f"No trained TensorFlow/Keras model found. Searched: {searched}"

    try:
        return tf.keras.models.load_model(model_file), None
    except Exception as exc:
        return None, f"Could not load TensorFlow model: {exc}"


def crop_center(image, margin=0.08):
    width, height = image.size
    left = int(width * margin)
    top = int(height * margin)
    right = int(width * (1 - margin))
    bottom = int(height * (1 - margin))
    return image.crop((left, top, right, bottom))


def crop_leaf_region(image):
    rgb = np.asarray(image.convert("RGB"))
    red = rgb[:, :, 0].astype(np.int16)
    green = rgb[:, :, 1].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)
    mask = (green > red + 8) & (green > blue + 4) & (green > 45)
    ys, xs = np.where(mask)
    if len(xs) < rgb.shape[0] * rgb.shape[1] * 0.03:
        return crop_center(image)

    pad_x = int((xs.max() - xs.min() + 1) * 0.12)
    pad_y = int((ys.max() - ys.min() + 1) * 0.12)
    left = max(int(xs.min()) - pad_x, 0)
    top = max(int(ys.min()) - pad_y, 0)
    right = min(int(xs.max()) + pad_x + 1, image.size[0])
    bottom = min(int(ys.max()) + pad_y + 1, image.size[1])
    return image.crop((left, top, right, bottom))


def preprocess_view(image, target_size=IMAGE_SIZE):
    image = image.convert("RGB").resize(target_size, Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32)
    return arr


def preprocess_image(image, target_size=IMAGE_SIZE):
    image = image.convert("RGB")
    views = [
        image,
        crop_center(image),
        crop_leaf_region(image),
    ]
    return np.stack([preprocess_view(view, target_size) for view in views], axis=0)


def prettify_label(label):
    if "___" in label:
        plant, condition = label.split("___", 1)
    else:
        plant, condition = label, ""

    plant = plant.replace("_", " ").replace(",", ", ").strip()
    condition = condition.replace("_", " ").strip() or "Unknown"
    return plant, condition


def plant_key(label):
    plant, _ = prettify_label(label)
    return plant.lower().replace(" ", "").replace(",", "")


def available_plants():
    plants = sorted({
        prettify_label(label)[0]
        for label in CLASSES
        if label != BACKGROUND_CLASS and "___" in label
    })
    return plants


def infer_plant_from_filename(filename):
    text = (filename or "").lower().replace("_", " ").replace("-", " ")
    for token, plant in PLANT_ALIASES.items():
        if token in text:
            return plant
    return None


def predict_disease(image, selected_plant=None):
    model, error = load_tf_model()
    if error:
        raise RuntimeError(error)

    if len(CLASSES) != int(model.output_shape[-1]):
        raise RuntimeError(
            f"Class list has {len(CLASSES)} labels, but the model outputs {int(model.output_shape[-1])} scores. "
            "Regenerate model_metadata.json or fix DEFAULT_CLASSES to match the training folder order."
        )

    if hasattr(model, "input_shape") and len(model.input_shape) >= 3:
        target_size = tuple(int(dim) for dim in model.input_shape[1:3])
    else:
        target_size = IMAGE_SIZE
    
    probs = model.predict(preprocess_image(image, target_size), verbose=0).mean(axis=0)
    display_probs = probs.copy()
    background_idx = CLASSES.index(BACKGROUND_CLASS) if BACKGROUND_CLASS in CLASSES else None
    if background_idx is not None:
        display_probs[background_idx] = 0.0

    if selected_plant:
        selected_key = selected_plant.lower().replace(" ", "").replace(",", "")
        for idx, label in enumerate(CLASSES):
            if label == BACKGROUND_CLASS or plant_key(label) != selected_key:
                display_probs[idx] = 0.0

    disease_total = float(display_probs.sum())
    if disease_total <= 0:
        plant_message = f" for {selected_plant}" if selected_plant else ""
        raise RuntimeError(f"The model did not return any usable disease scores{plant_message}.")

    display_probs = display_probs / disease_total
    top_indices = [
        idx for idx in np.argsort(display_probs)[::-1]
        if CLASSES[int(idx)] != BACKGROUND_CLASS
    ][:5]
    return [(CLASSES[int(i)], float(display_probs[int(i)])) for i in top_indices]


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
:root {
    --bg:#f4f7f1;
    --surface:#ffffff;
    --ink:#102018;
    --muted:#596c60;
    --line:rgba(16,32,24,.15);
    --leaf:#2f7d32;
    --leaf-dark:#173622;
    --select:#255fc7;
}

::selection {
    background:var(--select) !important;
    color:#ffffff !important;
}

::-moz-selection {
    background:var(--select) !important;
    color:#ffffff !important;
}

.stApp {
    background:var(--bg);
    color:var(--ink);
}

[data-testid="stHeader"] {
    background:transparent;
}

.block-container {
    padding:2rem 3rem 3rem;
    max-width:1280px;
}

h1, h2, h3, h4, h5, h6, p, label, span {
    color:var(--ink);
    letter-spacing:0;
}

.hero {
    min-height:310px;
    display:flex;
    align-items:flex-end;
    padding:32px;
    border-radius:8px;
    background:
        linear-gradient(90deg, rgba(8,22,12,.9), rgba(8,22,12,.42)),
        url('https://images.unsplash.com/photo-1464226184884-fa280b87c399?auto=format&fit=crop&w=1600&q=80');
    background-size:cover;
    background-position:center;
    box-shadow:0 22px 60px rgba(21,33,22,.18);
    margin-bottom:16px !important;
}

.hero h1 {
    color:white !important;
    font-size:clamp(2.4rem, 5vw, 5rem);
    line-height:.95;
    margin:0 0 10px;
}

.hero p {
    color:#eef8e8 !important;
    max-width:650px;
    font-size:1.05rem;
    font-weight:650;
}

.badge {
    display:inline-flex;
    padding:8px 14px;
    border-radius:999px;
    background:rgba(255,255,255,.18);
    color:white !important;
    font-weight:800;
    margin-bottom:14px;
    backdrop-filter:blur(8px);
}

[data-testid="stVerticalBlock"] {
    gap:.65rem !important;
}

[data-testid="stHorizontalBlock"] {
    align-items:stretch;
}

[data-testid="stWidgetLabel"],
[data-testid="stMarkdownContainer"]:empty {
    display:none !important;
}

.panel {
    background:var(--surface);
    border:1px solid var(--line);
    border-radius:8px;
    padding:16px;
    box-shadow:0 12px 32px rgba(21,33,22,.08);
    margin-top:0 !important;
    margin-bottom:0 !important;
}

.section-label {
    color:#0f6b24 !important;
    font-weight:900;
    text-transform:uppercase;
    font-size:.78rem;
    letter-spacing:.04rem;
    margin:0 0 6px;
}

[data-testid="stFileUploader"] {
    margin-top:0 !important;
}

[data-testid="stFileUploader"] section {
    border:1.5px dashed rgba(47,125,50,.55) !important;
    background:#fbfdf9 !important;
    border-radius:8px !important;
    padding:14px !important;
}

[data-testid="stFileUploader"] section * {
    color:#102018 !important;
}

[data-testid="stFileUploader"] button,
[data-testid="stFileUploader"] button[kind="secondary"] {
    background:#eef3ec !important;
    color:#102018 !important;
    border:1px solid rgba(16,32,24,.12) !important;
    border-radius:8px !important;
    font-weight:850 !important;
    box-shadow:none !important;
}

[data-testid="stFileUploader"] button:hover {
    background:#dfeadd !important;
    color:#102018 !important;
    border-color:rgba(47,125,50,.35) !important;
}

[data-testid="stFileUploader"] svg {
    color:#102018 !important;
    fill:#102018 !important;
}

.stButton>button {
    background:var(--leaf-dark)!important;
    color:white!important;
    border:0!important;
    border-radius:8px!important;
    min-height:46px;
    font-weight:900!important;
}

.stButton>button:hover {
    background:var(--leaf)!important;
    color:white!important;
}

.metric-card {
    background:#fbfdf9;
    border:1px solid var(--line);
    border-left:5px solid var(--leaf);
    border-radius:8px;
    padding:18px;
    min-height:106px;
}

.metric-label {
    color:#0f6b24 !important;
    font-size:.82rem;
    font-weight:900;
    text-transform:uppercase;
    margin:0 0 10px;
}

.metric-value {
    color:#102018 !important;
    font-size:2.05rem;
    line-height:1.05;
    font-weight:950;
    margin:0;
    overflow-wrap:anywhere;
}

.result-title {
    font-size:1.4rem;
    font-weight:900;
    margin:0;
    color:var(--leaf-dark) !important;
}

.result-subtitle {
    color:var(--muted) !important;
    margin:4px 0 0;
}

.status-warning {
    background:#fff8bd;
    color:#213427 !important;
    border-left:5px solid #d59600;
    border-radius:8px;
    padding:16px 18px;
    font-weight:750;
    line-height:1.55;
    overflow-wrap:anywhere;
}

.status-success {
    background:#e5f4e2;
    color:#153a1f !important;
    border-left:5px solid var(--leaf);
    border-radius:8px;
    padding:16px 18px;
    font-weight:750;
    line-height:1.55;
    overflow-wrap:anywhere;
}

.path-text {
    color:#213427 !important;
    font-size:1rem;
    line-height:1.6;
    overflow-wrap:anywhere;
}

.footer {
    color:var(--muted) !important;
    font-size:.9rem;
    padding-top:24px;
}

[data-baseweb="tab-list"] {
    border-bottom:2px solid var(--line) !important;
    background:transparent !important;
}

[role="tab"] {
    color:#102018 !important;
    font-weight:850 !important;
    font-size:1rem !important;
}

[role="tab"][aria-selected="true"] {
    color:#0f6b24 !important;
    border-bottom:3px solid #ef3f3f !important;
}

[role="tablist"] > button {
    background:transparent !important;
}

[data-testid="stAlert"],
.stAlert {
    color:#213427 !important;
}

[data-testid="stAlert"] *,
.stAlert * {
    color:#213427 !important;
    font-weight:650 !important;
}

input, textarea {
    color:var(--ink) !important;
    background:#ffffff !important;
}

.stProgress div {
    color:var(--ink) !important;
}

@media (max-width:700px) {
    .block-container {
        padding:1rem;
    }

    .hero {
        min-height:240px;
        padding:22px;
    }

    .metric-value {
        font-size:1.55rem;
    }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div>
        <div class="badge">TensorFlow fine-tuned disease classification</div>
        <h1>LeafScan</h1>
        <p>Upload a plant leaf image, run the trained Keras model, and review disease predictions with confidence scores.</p>
    </div>
</div>
""", unsafe_allow_html=True)

tab_detect, tab_system = st.tabs(["Disease Detection", "System"])

with tab_detect:
    left, right = st.columns([1.05, .95], gap="large")

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<p class="section-label">Image upload</p>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload JPG, JPEG, or PNG",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        inferred_plant = infer_plant_from_filename(uploaded_file.name) if uploaded_file else None
        plant_choices = ["Auto"] + available_plants()
        plant_index = plant_choices.index(inferred_plant) if inferred_plant in plant_choices else 0
        selected_plant_choice = st.selectbox(
            "Plant type",
            plant_choices,
            index=plant_index,
            help="Pick the crop to prevent cross-crop labels like corn for tomato or potato leaves.",
        )
        selected_plant = None if selected_plant_choice == "Auto" else selected_plant_choice
        if selected_plant is None:
            selected_plant = inferred_plant

        analyze = st.button("Run disease prediction", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB") if uploaded_file else None
        if image:
            st.image(image, caption="Uploaded leaf image", use_container_width=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<p class="section-label">Model output</p>', unsafe_allow_html=True)

        if analyze:
            if image is None:
                st.warning("Upload an image before running prediction.")
            else:
                try:
                    with st.spinner("Running TensorFlow inference..."):
                        predictions = predict_disease(image, selected_plant)

                    top_label, top_conf = predictions[0]
                    plant, condition = prettify_label(top_label)

                    st.markdown(
                        f'<div class="metric-card"><p class="result-title">{condition}</p>'
                        f'<p class="result-subtitle">Plant: {plant} | Confidence: {top_conf * 100:.2f}%</p></div>',
                        unsafe_allow_html=True,
                    )

                    st.write("")
                    if selected_plant:
                        st.caption(f"Filtered to {selected_plant} classes.")
                    st.write("Top predictions")

                    for label, conf in predictions:
                        plant_name, condition_name = prettify_label(label)
                        st.progress(conf, text=f"{plant_name} - {condition_name}: {conf * 100:.2f}%")

                except Exception as exc:
                    st.error(str(exc))
        else:
            st.info("Upload a leaf and run prediction to see live results here.")

        st.markdown('</div>', unsafe_allow_html=True)

with tab_system:
    model_candidates = [MODEL_PATH, FALLBACK_KERAS_MODEL_PATH, FALLBACK_H5_MODEL_PATH]
    model_source = next((path for path in model_candidates if path.exists()), MODEL_PATH)

    if not TF_AVAILABLE:
        system_message = "TensorFlow is not installed in this environment."
        system_ok = False
    elif not model_source.exists():
        searched = ", ".join(str(path) for path in model_candidates)
        system_message = f"No trained TensorFlow/Keras model found. Searched: {searched}"
        system_ok = False
    else:
        system_message = f"Model file linked: {model_source}. It will load when you run prediction."
        system_ok = True

    c1, c2, c3 = st.columns(3)

    c1.markdown(
        f'<div class="metric-card"><p class="metric-label">Classes</p>'
        f'<p class="metric-value">{len(CLASSES)}</p></div>',
        unsafe_allow_html=True,
    )

    c2.markdown(
        f'<div class="metric-card"><p class="metric-label">Framework</p>'
        f'<p class="metric-value">{"TensorFlow" if TF_AVAILABLE else "Unavailable"}</p></div>',
        unsafe_allow_html=True,
    )

    c3.markdown(
        f'<div class="metric-card"><p class="metric-label">Input size</p>'
        f'<p class="metric-value">{IMAGE_SIZE[0]} x {IMAGE_SIZE[1]}</p></div>',
        unsafe_allow_html=True,
    )

    st.write("")

    status_class = "status-success" if system_ok else "status-warning"
    st.markdown(f'<div class="{status_class}">{system_message}</div>', unsafe_allow_html=True)

    st.markdown(f'<p class="path-text"><strong>Dataset:</strong> {DATASET_PATH}</p>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="path-text"><strong>Metadata source:</strong> '
        f'{METADATA.get("dataset_path", METADATA.get("source", str(METADATA_PATH)))}</p>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="footer">LeafScan | TensorFlow frontend relinked, backend app surface preserved.</div>',
    unsafe_allow_html=True,
)
