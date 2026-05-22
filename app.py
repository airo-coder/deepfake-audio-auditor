import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import joblib
import os

st.set_page_config(page_title="Deepfake Audio Auditor Dashboard", layout="wide")

# --- CUSTOM CACHED LOADERS FOR MAXIMUM PERFORMANCE ---
@st.cache_resource
def load_verification_system():
    # compile=False allows Keras to load the model architecture weights safely
    nn_brain = tf.keras.models.load_model("enhanced_attention_model_v5.keras", compile=False)
    feature_extractor = tf.keras.models.Model(inputs=nn_brain.input, outputs=nn_brain.get_layer('deep_features').output)
    xgb_judge = joblib.load("xgboost_refiner_v5.pkl")
    return feature_extractor, xgb_judge

# Ensure model binaries exist before executing loading structures
if os.path.exists("enhanced_attention_model_v5.keras") and os.path.exists("xgboost_refiner_v5.pkl"):
    feature_extractor, xgb_judge = load_verification_system()
else:
    st.error("🛑 Architecture Dependency Error: Model binary weights (.keras/.pkl) missing from directory root.")

# --- SIDEBAR STATISTICS PANEL ---
with st.sidebar:
    st.title("🛡️ Auditing Specifications")
    st.markdown("**Engine Version:** V5-Attention-Robust")
    st.markdown("---")
    st.subheader("📊 Empirically Validated Metrics")
    st.metric(label="System Testing Accuracy", value="95.0%")
    st.metric(label="Area Under Curve (ROC-AUC)", value="0.9848")
    st.metric(label="Target Human F1-Score", value="0.95")
    st.metric(label="Target AI F1-Score", value="0.95")
    st.markdown("---")
    st.caption("Developed for the University of Mindanao CS Thesis Defense 2026.")

# --- MAIN ACCOUNTABILITY PANEL ---
st.title("🎙️ Multi-Layer Audio Deepfake Authentication Dashboard")
st.write("Upload an audio sample to extract its 60-D dynamic acoustic signature and run neural validation.")

uploaded_file = st.file_uploader("Select Target Audio Sample (.wav, .mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    st.markdown("---")
    left_col, right_col = st.columns([1, 1])
    
    with left_col:
        st.subheader("🔍 Automated Security Diagnosis")
        
        try:
            # 1. Pipeline Data Extraction
            y, sr = librosa.load(uploaded_file, sr=16000)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            d1 = librosa.feature.delta(mfcc)
            d2 = librosa.feature.delta(mfcc, order=2)
            raw_features = np.vstack([mfcc, d1, d2])
            
            # Pad/Truncate to exactly 130 frames to match Colab tensor input sizes
            if raw_features.shape[1] < 130:
                padded = np.pad(raw_features, pad_width=((0,0), (0, 130-raw_features.shape[1])), mode='constant')
            else:
                padded = raw_features[:, :130]
            
            input_tensor = np.expand_dims(padded.T, axis=0) # Shape format: (1, 130, 6
