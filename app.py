import streamlit as st
import numpy as np
import re
import time
import pickle

# 1. Page Configuration
st.set_page_config(
    page_title="Lumina AI - Detector & Humanizer", 
    page_icon="🔮",
    layout="wide"
)

# 2. Pipeline Loader
@st.cache_resource
def load_trained_engine():
    try:
        with open("detector_model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except Exception:
        return None

ml_model = load_trained_engine()

# 3. Organic Humanizer Engine
def execute_humanizer(text):
    replacements = {
        " furthermore, ": " also, ",
        " moreover, ": " plus, ",
        " therefore, ": " so, ",
        " consequently, ": " as a result, ",
        " in conclusion, ": " long story short, "
    }
    humanized = text
    for target, replacement in replacements.items():
        pattern = re.compile(re.escape(target), re.IGNORECASE)
        humanized = pattern.sub(replacement, humanized)
    return humanized

# 4. Main UI Layout
st.title("🔮 Lumina AI Engine Pro")
st.caption("Enterprise AI Text Detection & Organic Humanization")
st.markdown("---")

with st.sidebar:
    st.header("📖 System Specs")
    if ml_model is not None:
        st.success("🤖 High-Vocabulary ML Core: Active")
    else:
        st.warning("⚠️ Fallback active (detector_model.pkl not loaded)")

left_col, right_col = st.columns(2)

with left_col:
    st.subheader("📥 Input Console")
    user_input = st.text_area("Paste your text here to process:", height=320, placeholder="Enter text here...")
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        detect_clicked = st.button("🔍 Run Forensic Scan", use_container_width=True, type="primary")
    with btn_col2:
        humanize_clicked = st.button("✨ Apply Organic Humanizer", use_container_width=True)

with right_col:
    st.subheader("📤 Neural Output Matrix")
    if detect_clicked:
        if not user_input.strip():
            st.warning("⚠️ Enter text to initiate diagnostic systems.")
        else:
            with st.spinner("Scanning structural & vocabulary matrices..."):
                time.sleep(0.3)
                if ml_model is not None:
                    try:
                        prediction_proba = ml_model.predict_proba([user_input])[0]
                        raw_ai_prob = prediction_proba[1] * 100
                        
                        # Soft-smoothing calibration zone to minimize layout/heading false positives
                        if raw_ai_prob > 80:
                            ai_probability = int(raw_ai_prob)
                        elif raw_ai_prob > 40:
                            ai_probability = int(raw_ai_prob * 0.82)
                        else:
                            ai_probability = int(raw_ai_prob * 0.55)
                            
                        # Display Results with Adjusted Threshold Boundaries
                        if ai_probability > 75:
                            st.error(f"🚨 AI Profile Confirmed ({ai_probability}% Probability)")
                            st.progress(ai_probability)
                        elif 45 <= ai_probability <= 75:
                            st.warning(f"⚠️ Mixed Phrasing / Uncertain Structural Signature ({ai_probability}% AI Bias)")
                            st.progress(ai_probability)
                        else:
                            st.success(f"✅ Verified Organic Human Writing ({100 - ai_probability}% Certainty)")
                            st.progress(ai_probability)
                            
                    except Exception as eval_error:
                        st.error(f"Execution Error: {eval_error}")
                else:
                    st.error("Model engine offline. Please check detector_model.pkl status.")
                    
    elif humanize_clicked:
        if not user_input.strip():
            st.warning("⚠️ Enter text to process humanization modifications.")
        else:
            st.write("✨ **Humanized Output:**")
            st.text_area("Copy Output:", value=execute_humanizer(user_input), height=220)
    else:
        st.info("System idle. Initialize an input operation to populate telemetry.")