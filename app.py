"""Handwritten Notes to Digital Text Converter - Streamlit Application.

An end-to-end Optical Character Recognition (OCR) web application that
converts handwritten notes into digitized, editable text using OpenCV
preprocessing pipelines and deep learning EasyOCR models.
"""

import os
import json
import time
import datetime
import cv2
from PIL import Image
import numpy as np
import streamlit as st

# Import core computer vision and OCR utilities
from utils.image_utils import (
    load_image,
    preprocess_image,
)
from utils.ocr_utils import (
    get_ocr_reader,
    run_ocr,
)

# ---------------------------------------------------------------------------
# 1. Page Configuration & Styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Handwritten Notes to Digital Text Converter",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Modern, corporate styling - Strictly zero emojis, dark/light theme adaptive
st.markdown(
    """
    <style>
    /* Global layout & typography */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1200px;
    }
    
    /* Header styling */
    .app-header-title {
        font-size: 1.7rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
    }
    .app-header-subtitle {
        font-size: 0.95rem;
        opacity: 0.8;
        margin-bottom: 1.5rem;
        line-height: 1.45;
    }
    
    /* Sidebar info card */
    .sidebar-card {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 8px;
        padding: 0.9rem;
        margin-bottom: 1rem;
    }
    .sidebar-card-title {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 700;
        opacity: 0.75;
        margin-bottom: 0.35rem;
    }
    .sidebar-card-name {
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    .sidebar-card-desc {
        font-size: 0.82rem;
        opacity: 0.8;
        line-height: 1.35;
        margin: 0;
    }
    
    /* Metrics display card */
    .metric-container {
        display: flex;
        gap: 1rem;
        margin-top: 0.75rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        flex: 1;
        background: rgba(128, 128, 128, 0.07);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
        padding: 0.85rem 1rem;
        text-align: center;
    }
    .metric-card-title {
        font-size: 0.72rem;
        text-transform: uppercase;
        opacity: 0.75;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .metric-card-value {
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 0.25rem;
    }
    
    /* Ground truth comparison box */
    .ground-truth-box {
        background: rgba(14, 165, 233, 0.08);
        border: 1px solid rgba(14, 165, 233, 0.3);
        border-radius: 8px;
        padding: 0.85rem 1rem;
        margin-top: 0.75rem;
        margin-bottom: 1rem;
    }
    .ground-truth-title {
        font-size: 0.78rem;
        font-weight: 700;
        color: #0284c7;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .ground-truth-text {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.92rem;
        margin-top: 0.3rem;
        margin-bottom: 0;
    }
    
    /* Professional footer */
    .app-footer {
        margin-top: 3.5rem;
        padding-top: 1.5rem;
        border-top: 1px solid rgba(128, 128, 128, 0.2);
        text-align: center;
        font-size: 0.85rem;
        opacity: 0.85;
    }
    .app-footer a {
        color: #38bdf8;
        text-decoration: none;
        font-weight: 500;
        margin: 0 0.25rem;
    }
    .app-footer a:hover {
        text-decoration: underline;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# 2. Resource Caching
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_cached_ocr_reader():
    """Initializes and caches the EasyOCR Reader in memory."""
    return get_ocr_reader(languages=["en"], gpu=False)


@st.cache_data
def load_sample_dataset_metadata():
    """Loads IAM handwriting sample metadata from disk."""
    metadata_path = os.path.join("dataset", "metadata.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


# ---------------------------------------------------------------------------
# 3. Sidebar Configuration (Clean, No Confusing Options)
# ---------------------------------------------------------------------------

# Use the single best preprocessing pipeline verified by empirical benchmarking (53.75% accuracy)
OPTIMIZED_PIPELINE = "enhanced_grayscale"

with st.sidebar:
    st.subheader("System Configuration")
    
    st.markdown(
        """
        <div class="sidebar-card">
            <div class="sidebar-card-title">Active Vision Pipeline</div>
            <div class="sidebar-card-name">Enhanced Grayscale (Bilateral + CLAHE)</div>
            <p class="sidebar-card-desc">
                Automatically applies edge-preserving noise filtering and contrast equalization. 
                Optimized specifically for handwritten ink on paper.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("**Architecture Overview**")
    st.markdown("- **Text Detection**: CRAFT Neural Network")
    st.markdown("- **Text Recognition**: CRNN + ResNet + BiLSTM")
    st.markdown("- **Framework**: EasyOCR / PyTorch / OpenCV")
    st.markdown("- **Benchmark Dataset**: IAM Handwriting Database (7 samples)")
    
    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 0.82rem; opacity: 0.85; line-height: 1.5;">
            <strong>Developer</strong><br>
            Tuhid Khan<br>
            <a href="https://github.com/TuhidKhan" target="_blank" style="color: #38bdf8; text-decoration: none;">GitHub</a> &nbsp;|&nbsp;
            <a href="https://www.linkedin.com/in/tuhid-khan/" target="_blank" style="color: #38bdf8; text-decoration: none;">LinkedIn</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# 4. Main Interface Header
# ---------------------------------------------------------------------------

st.markdown('<div class="app-header-title">Handwritten Notes to Digital Text Converter</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-header-subtitle">'
    'Deep learning Optical Character Recognition system with automated OpenCV preprocessing. '
    'Upload a handwritten note or select a curated IAM benchmark sample to extract digital text.'
    '</div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# 5. Input Image Selection (Upload vs. IAM Dataset)
# ---------------------------------------------------------------------------

input_source_tab = st.radio(
    "Select Input Source",
    options=["Select IAM Dataset Sample", "Upload Image File"],
    horizontal=True,
    label_visibility="collapsed",
)

input_image_raw = None
selected_sample_meta = None
source_name = "handwritten_note"

if input_source_tab == "Upload Image File":
    uploaded_file = st.file_uploader(
        "Upload handwritten note (PNG, JPG, JPEG, WEBP, BMP)",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        help="Supports standard image formats up to 15 MB.",
    )
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        input_image_raw = load_image(file_bytes)
        source_name = os.path.splitext(uploaded_file.name)[0]

else:
    samples = load_sample_dataset_metadata()
    if samples:
        sample_options = {
            f"{s['id'].upper()}: {s['ground_truth_text'][:45]}...": s
            for s in samples
        }
        chosen_label = st.selectbox(
            "Select IAM handwriting benchmark sample",
            options=list(sample_options.keys()),
        )
        selected_sample_meta = sample_options[chosen_label]
        sample_img_path = os.path.join("dataset", "input_images", selected_sample_meta["image_file"])
        if os.path.exists(sample_img_path):
            input_image_raw = load_image(sample_img_path)
            source_name = selected_sample_meta["id"]
    else:
        st.warning("No IAM dataset samples found in the dataset directory.")


# ---------------------------------------------------------------------------
# 6. Interactive Extraction & Visual Inspection
# ---------------------------------------------------------------------------

if input_image_raw is not None:
    # Execute computer vision preprocessing pipeline (Enhanced Grayscale)
    orig_bgr, proc_bgr = preprocess_image(input_image_raw, pipeline_mode=OPTIMIZED_PIPELINE)
    
    # Convert BGR to RGB for accurate rendering in Streamlit
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB) if len(orig_bgr.shape) == 3 else orig_bgr
    proc_rgb = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2RGB) if len(proc_bgr.shape) == 3 else proc_bgr
    
    # Prominent Action Trigger placed directly with input selection
    st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
    extract_col1, extract_col2 = st.columns([1, 2])
    with extract_col1:
        extract_button = st.button("Extract Handwritten Text", type="primary", use_container_width=True)
    with extract_col2:
        st.caption("Click to execute deep learning OCR inference and character recognition.")

    # Execution logic on click
    if extract_button:
        with st.spinner("Processing image and executing neural network character recognition..."):
            start_time = time.time()
            
            # Ensure model is initialized
            reader = load_cached_ocr_reader()
            
            # Run OCR inference and reading-order postprocessing
            result = run_ocr(
                image=proc_bgr,
                reader=reader,
                save_output=True,
                base_name=source_name,
            )
            elapsed_time = round(time.time() - start_time, 3)
            
            # Store results in session state
            st.session_state["ocr_result"] = result
            st.session_state["elapsed_time"] = elapsed_time
            st.session_state["source_name"] = source_name
            st.session_state["selected_sample_meta"] = selected_sample_meta

    # Image Inspection Previews (Original vs. Preprocessed)
    st.markdown("#### Image Inspection")
    col_orig, col_proc = st.columns(2)
    
    with col_orig:
        st.caption("Original Input Image")
        st.image(orig_rgb, use_container_width=True)
        
    with col_proc:
        st.caption("Preprocessed Image (Enhanced Grayscale)")
        st.image(proc_rgb, use_container_width=True)
    
    # -----------------------------------------------------------------------
    # 7. Display Results & Transcribed Output
    # -----------------------------------------------------------------------
    if "ocr_result" in st.session_state and st.session_state.get("source_name") == source_name:
        res = st.session_state["ocr_result"]
        elapsed = st.session_state.get("elapsed_time", 0.0)
        curr_sample_meta = st.session_state.get("selected_sample_meta")
        
        st.markdown("#### Recognition Results")
        
        # Metrics summary cards
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-card">
                    <div class="metric-card-title">Execution Time</div>
                    <div class="metric-card-value">{elapsed} s</div>
                </div>
                <div class="metric-card">
                    <div class="metric-card-title">Average Confidence</div>
                    <div class="metric-card-value">{int(res['average_confidence'] * 100)}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-card-title">Words Detected</div>
                    <div class="metric-card-value">{res['word_count']}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Ground truth comparison (if sample is selected)
        if curr_sample_meta:
            gt_text = curr_sample_meta.get("ground_truth_text", "")
            st.markdown(
                f"""
                <div class="ground-truth-box">
                    <div class="ground-truth-title">Ground-Truth Reference (IAM Database)</div>
                    <p class="ground-truth-text">{gt_text}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        # Editable Extracted Text Area
        extracted_text_value = st.text_area(
            "Extracted Digital Text (Editable)",
            value=res["text"],
            height=140,
            help="You can review and edit the transcribed text directly.",
        )
        
        # Actions: Download and Disk Persistence Notification
        col_dl, col_info = st.columns([1, 2])
        
        with col_dl:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"extracted_{st.session_state.get('source_name', 'note')}_{timestamp}.txt"
            
            st.download_button(
                label="Download Text File (.txt)",
                data=extracted_text_value,
                file_name=filename,
                mime="text/plain",
                use_container_width=True,
            )
            
        with col_info:
            saved_path = res.get("saved_file")
            if saved_path and os.path.exists(saved_path):
                st.info(f"Output automatically archived to: {saved_path}")

else:
    st.info("Select an IAM benchmark sample above or upload an image to begin OCR processing.")


# ---------------------------------------------------------------------------
# 8. Professional Footer
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="app-footer">
        Developed by <strong>Tuhid Khan</strong> &nbsp;|&nbsp; 
        <a href="https://github.com/TuhidKhan" target="_blank" rel="noopener noreferrer">GitHub</a> &nbsp;|&nbsp; 
        <a href="https://www.linkedin.com/in/tuhid-khan/" target="_blank" rel="noopener noreferrer">LinkedIn</a>
    </div>
    """,
    unsafe_allow_html=True,
)
