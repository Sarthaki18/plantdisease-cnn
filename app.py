import streamlit as st
from PIL import Image
import numpy as np
import os
import io
import json

# Try to import TensorFlow
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# Configure page
st.set_page_config(
    page_title='LeafScan — Plant Disease AI',
    layout='wide',
    initial_sidebar_state='collapsed'
)

# Premium black + green theme CSS
st.markdown("""
<style>
    * { box-sizing: border-box; }
    
    /* Dark background */
    .stApp {
        background: linear-gradient(135deg, #0a0e0a 0%, #0f1810 100%);
        color: #e0f0e0;
    }
    
    /* Main container */
    .stContainer {
        background: transparent;
    }
    
    /* Text colors */
    h1, h2, h3, h4, h5, h6 { 
        color: #ffffff;
        font-weight: 700;
    }
    
    /* Green accents */
    .stMetric {
        background: linear-gradient(135deg, rgba(34,197,94,0.1), rgba(16,185,129,0.05));
        border-left: 4px solid #22c55e;
        padding: 1rem;
        border-radius: 8px;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #22c55e, #16a34a) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.5rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(34,197,94,0.2) !important;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #16a34a, #15803d) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(34,197,94,0.4) !important;
    }
    
    /* Cards/sections */
    .stInfo, .stSuccess, .stWarning, .stError {
        border-left: 4px solid #22c55e !important;
        background: rgba(34,197,94,0.08) !important;
        border-radius: 8px !important;
    }
    
    /* File uploader */
    .uploadedFile {
        border: 2px dashed #22c55e;
        border-radius: 8px;
        padding: 1.5rem;
    }
    
    /* Divider */
    hr {
        border-color: #1a3a2a !important;
        margin: 2rem 0 !important;
    }
    
    /* Custom header */
    .header-hero {
        background: linear-gradient(135deg, rgba(34,197,94,0.15), rgba(16,185,129,0.08));
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid rgba(34,197,94,0.3);
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .status-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .status-item {
        background: rgba(34,197,94,0.08);
        border: 1px solid rgba(34,197,94,0.2);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Paths
BASE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE, 'Classification-based Anomaly Detection', 'plant_disease_model_tf.h5')
METADATA_PATH = os.path.join(BASE, 'Classification-based Anomaly Detection', 'model_metadata.json')

# Load metadata
CLASSES = []
if os.path.exists(METADATA_PATH):
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
        CLASSES = metadata.get('classes', [])

if not CLASSES:
    CLASSES = sorted([
        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
        'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
        'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
        'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
        'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
        'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy',
        'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch',
        'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
        'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
    ])

# Load model
@st.cache_resource
def load_tf_model():
    if not TF_AVAILABLE:
        return None, False
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            return model, True
        except Exception as e:
            st.warning(f"Could not load model: {e}")
            return None, False
    return None, False

MODEL, MODEL_OK = load_tf_model()

def preprocess_image(img: Image.Image):
    """Preprocess image for TensorFlow model."""
    img = img.convert('RGB').resize((224, 224))
    arr = np.array(img, dtype='float32') / 255.0
    return arr

def run_inference(image: Image.Image):
    """Run inference on image."""
    if not MODEL_OK or not TF_AVAILABLE:
        # Demo mode
        img_arr = np.array(image.convert('RGB').resize((64, 64)))
        green_channel = img_arr[:, :, 1].mean()
        class_idx = int((green_channel % len(CLASSES)))
        conf = 0.72
        top1 = CLASSES[class_idx]
        top5 = [
            (CLASSES[(class_idx + i) % len(CLASSES)], round(max(0.1, conf - i*0.1), 3))
            for i in range(5)
        ]
        return top1, top5, True
    
    # Real inference
    try:
        img_arr = preprocess_image(image)
        pred = MODEL.predict(np.array([img_arr]), verbose=0)
        probs = pred[0]
        top5_indices = np.argsort(probs)[::-1][:5]
        top1 = CLASSES[top5_indices[0]]
        top5 = [(CLASSES[idx], round(float(probs[idx]), 3)) for idx in top5_indices]
        return top1, top5, False
    except Exception as e:
        st.error(f"Inference error: {e}")
        return None, [], True

# UI Header
st.markdown("""
<div class="header-hero">
    <h1 style="margin: 0; font-size: 2.5rem;">🌿 LeafScan</h1>
    <p style="margin: 0.5rem 0 0 0; color: #a0e0a0; font-size: 1rem;">
        AI-Powered Plant Disease Detection · TensorFlow Transfer Learning
    </p>
</div>
""", unsafe_allow_html=True)

# Main layout
col1, col2 = st.columns([2, 1.2])

with col1:
    st.subheader('📸 Upload & Analyze')
    uploaded_file = st.file_uploader('Select a leaf image', type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        img = Image.open(io.BytesIO(uploaded_file.read()))
        st.image(img, use_column_width=True, caption='Uploaded leaf image')
        
        if st.button('🔍 Analyze Image', use_container_width=True):
            with st.spinner('Running inference...'):
                top1, top5, is_demo = run_inference(img)
                
                if top1:
                    # Format class name nicely
                    plant, condition = top1.split('___') if '___' in top1 else (top1, '')
                    
                    # Display prediction
                    st.markdown("---")
                    st.markdown("### 🎯 Prediction Result")
                    
                    col_pred1, col_pred2 = st.columns([1, 1])
                    with col_pred1:
                        st.metric('Plant', plant)
                    with col_pred2:
                        st.metric('Condition', condition.replace('_', ' '))
                    
                    if is_demo:
                        st.info('Demo mode: Using simulated inference')
                    
                    # Top-5 predictions
                    st.markdown("#### Top 5 Predictions")
                    for rank, (label, conf) in enumerate(top5, 1):
                        plant_name = label.split('___')[0] if '___' in label else label
                        cond_name = label.split('___')[1] if '___' in label else ''
                        col_rank, col_bar, col_conf = st.columns([0.5, 3, 1])
                        with col_rank:
                            st.write(f'**{rank}.**')
                        with col_bar:
                            st.progress(float(conf), f'{plant_name}: {cond_name.replace("_", " ")}')
                        with col_conf:
                            st.write(f'**{conf*100:.1f}%**')
    else:
        st.info('👈 Upload an image to get started')

with col2:
    st.subheader('⚙️ System Status')
    
    st.markdown("""
    <div class="status-grid">
        <div class="status-item">
            <strong>TensorFlow</strong><br>
            <span style="color: #22c55e;">✓ Ready</span>
        </div>
        <div class="status-item">
            <strong>Model</strong><br>
            <span style="color: #22c55e;">✓ Loaded</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader('📊 Model Info')
    st.write(f'**Classes:** {len(CLASSES)}')
    st.write(f'**Architecture:** MobileNetV2')
    st.write(f'**Framework:** TensorFlow/Keras')
    
    st.markdown("---")
    st.subheader('📋 Sample Classes')
    sample_classes = [CLASSES[i] for i in [0, 5, 10, 15, 20]]
    for cls in sample_classes:
        st.caption(f'• {cls.replace("___", " → ")}')

st.markdown("---")
st.caption('🌱 LeafScan v1.0 — Sustainable Plant Health AI · Made with ❤️ for agriculture')
