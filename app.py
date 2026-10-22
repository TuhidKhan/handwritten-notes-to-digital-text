"""Handwritten Notes to Digital Text Converter - Streamlit Application.

An end-to-end Optical Character Recognition (OCR) web application that
converts handwritten notes into digitized, editable text using OpenCV
preprocessing pipelines and deep learning EasyOCR models.
"""

import os
import json
import time
import datetime
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

# Corporate, clean styling - Strictly zero emojis, professional typography
st.markdown(
    """
    <style>
    /* Global layout & typography */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2.5rem;
        max-width: 1200px;
    }
    
    /* Header styling */
    .app-header-title {
        font-size: 1.65rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
    }
    .app-header-subtitle {
        font-size: 0.95rem;
        color: #64748b;
        margin-bottom: 1.5rem;
        line-height: 1.4;
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
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        text-align: center;
    }
    .metric-card-title {
        font-size: 0.75rem;
        text-transform: uppercase;
        color: #64748b;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .metric-card-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 0.2rem;
    }
    
    /* Ground truth comparison box */
    .ground-truth-box {
        background-color: #f0f9ff;
        border: 1px solid #bae6fd;
        border-radius: 6px;
        padding: 0.85rem 1rem;
        margin-top: 0.75rem;
        margin-bottom: 1rem;
    }
    .ground-truth-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #0369a1;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .ground-truth-text {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.9rem;
        color: #0c4a6e;
        margin-top: 0.3rem;
        margin-bottom: 0;
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
# 3. Sidebar Configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.subheader("System Configuration")
    
    pipeline_mode_label = st.selectbox(
        "Preprocessing Pipeline",
        options=[
            "Enhanced Grayscale (Bilateral Filter + CLAHE) [Recommended]",
            "Otsu Binarization (Global Histogram Threshold)",
            "Adaptive Gaussian Binarization (Local Neighborhood)",
            "Raw (No Preprocessing)",
        ],
        index=0,
        help="Select a computer vision filter to optimize contrast and suppress paper grain.",
    )
    
    # Map friendly label to pipeline key
    pipeline_mode_map = {
        "Enhanced Grayscale (Bilateral Filter + CLAHE) [Recommended]": "enhanced_grayscale",
        "Otsu Binarization (Global Histogram Threshold)": "otsu_binarized",
        "Adaptive Gaussian Binarization (Local Neighborhood)": "adaptive_binarized",
        "Raw (No Preprocessing)": "raw",
    }
    selected_pipeline = pipeline_mode_map[pipeline_mode_label]
    
    st.markdown("---")
    st.markdown("**Architecture Overview**")
    st.markdown("- **Text Detection**: CRAFT Neural Network")
    st.markdown("- **Text Recognition**: CRNN + ResNet + BiLSTM")
    st.markdown("- **Framework**: EasyOCR / PyTorch / OpenCV")
    st.markdown("- **Dataset**: IAM Handwriting Database (7 samples)")
    
    st.markdown("---")
    st.caption("Handwritten Notes to Digital Text Converter | Open-Source Architecture")


# ---------------------------------------------------------------------------
# 4. Main Interface Header
# ---------------------------------------------------------------------------

st.markdown('<div class="app-header-title">Handwritten Notes to Digital Text Converter</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-header-subtitle">'
    'Deep learning Optical Character Recognition system with computer vision image preprocessing. '
    'Upload a handwritten note image or evaluate using built-in IAM dataset benchmark samples.'
    '</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 5. Input Image Selection (Upload vs. IAM Dataset)
# ---------------------------------------------------------------------------

input_source_tab = st.radio(
    "Select Input Source",
    options=["Upload Image File", "Select IAM Dataset Sample"],
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
# 6. Side-by-Side Visualizations & Preprocessing
# ---------------------------------------------------------------------------

if input_image_raw is not None:
    # Execute computer vision preprocessing pipeline
    orig_bgr, proc_bgr = preprocess_image(input_image_raw, pipeline_mode=selected_pipeline)
    
    # Convert BGR to RGB for correct Streamlit rendering
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB) if len(orig_bgr.shape) == 3 else orig_bgr
    proc_rgb = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2RGB) if len(proc_bgr.shape) == 3 else proc_bgr
    
    st.markdown("#### Image Inspection")
    col_orig, col_proc = st.columns(2)
    
    with col_orig:
        st.caption("Original Input Image")
        st.image(orig_rgb, use_container_width=True)
        
    with col_proc:
        st.caption(f"Preprocessed Image ({selected_pipeline})")
        st.image(proc_rgb, use_container_width=True)
    
    # -----------------------------------------------------------------------
    # 7. OCR Extraction Action
    # -----------------------------------------------------------------------
    st.markdown("#### OCR Text Extraction")
    extract_button = st.button("Extract Handwritten Text", type="primary", use_container_width=True)
    
    # Manage session state for results persistence across widget interactions
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
            
            # Store in session state
            st.session_state["ocr_result"] = result
            st.session_state["elapsed_time"] = elapsed_time
            st.session_state["source_name"] = source_name
            st.session_state["selected_sample_meta"] = selected_sample_meta
    
    # -----------------------------------------------------------------------
    # 8. Display Results & Metrics
    # -----------------------------------------------------------------------
    if "ocr_result" in st.session_state:
        res = st.session_state["ocr_result"]
        elapsed = st.session_state.get("elapsed_time", 0.0)
        curr_sample_meta = st.session_state.get("selected_sample_meta")
        
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
            height=160,
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
