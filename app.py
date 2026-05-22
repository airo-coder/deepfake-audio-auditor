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
    # Load the feature extracting brain and the judging refiner
    nn_brain = tf.keras.models.load_model("enhanced_attention_model_v5.keras")
    feature_extractor = tf.keras.models.Model(inputs=nn_brain.input, outputs=nn_brain.get_layer('deep_features').output)
    xgb_judge = joblib.load("xgboost_refiner_v5.pkl")
    return feature_extractor, xgb_judge

# Ensure files exist locally before attempting to run loading structures
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
            
            # Pad/Truncate to 130 frames
            if raw_features.shape[1] < 130:
                padded = np.pad(raw_features, pad_width=((0,0), (0, 130-raw_features.shape[1])), mode='constant')
            else:
                padded = raw_features[:, :130]
            
            input_tensor = np.expand_dims(padded.T, axis=0) # Shape: (1, 130, 60)
            
            # 2. Dual-Layer Prediction Execution
            deep_embeddings = feature_extractor.predict(input_tensor)
            ai_probability = xgb_judge.predict_proba(deep_embeddings)[0, 1]
            
            # 🔥 --- SECURITY THRESHOLD HARDENING --- 🔥
            HARDENED_THRESHOLD = 0.38
            
            if ai_probability >= HARDENED_THRESHOLD:
                st.error("🚨 **DIAGNOSIS REJECTED: HIGH ACCUMULATION OF SYNTHETIC ARTIFACTS**")
                verdict = "AI (Spoof)"
                confidence_score = ai_probability * 100
            else:
                st.success("✅ **DIAGNOSIS VERIFIED: BIOLOGICAL SPEECH SIGNATURE MATCHED**")
                verdict = "Human (Bonafide)"
                confidence_score = (1 - ai_probability) * 100
                
            st.metric(label="Decision Security Confidence", value=f"{confidence_score:.2f}%")
            
            # Visual Probability Bar
            st.write("Raw Algorithm Distribution Array:")
            st.progress(int(ai_probability * 100))
            st.caption("0% (Absolute Biological Human) <-----------------------------------------> 100% (Absolute Algorithmic Voice Synth)")
            
        except Exception as err:
            st.error(f"Execution Error: {err}")
            
    with right_col:
        st.subheader("📊 60-D Vocal Texture Mapping")
        try:
            # Generate your diagnostic texture mapping chart
            fig, ax = plt.subplots(figsize=(6, 4))
            img = librosa.display.specshow(raw_features, x_axis='time', cmap='magma', ax=ax)
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            ax.set_title("Extracted Spectral Fingerprint Vector Map")
            st.pyplot(fig)
            st.caption("💡 **Professor Reference Guide:** Inspect the vector flow. Sharp horizontal discontinuities and perfectly aligned lattice blocks show mathematical audio block synthesis. Dispersed, irregular energy patterns represent biological vocal chord variants.")
        except Exception as img_err:
            st.write("Unable to render feature chart visualization.")