"""Streamlit Reference Application for Handwritten Notes to Digital Text Converter.

This script implements the complete single-interface application mandated by
the project specification (Sections 5, 9, 10, and 13).
"""

import os
import time
import json
import streamlit as st
import numpy as np

from utils.image_utils import (
    load_image,
    preprocess_image,
)
from utils.ocr_utils import (
    get_ocr_reader,
    run_ocr,
)

# Page configuration
st.set_page_config(
    page_title="Handwritten Notes to Digital Text Converter",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional styling overrides
st.markdown(
    """
    <style>
    .main-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #555555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 4px;
        padding: 0.75rem 1rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title Header
st.markdown('<div class="main-header">Handwritten Notes to Digital Text Converter</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Optical Character Recognition system powered by OpenCV image preprocessing and EasyOCR.</div>',
    unsafe_allow_html=True,
)

# Sidebar Controls
st.sidebar.header("Configuration")

# Input selection method
input_mode = st.sidebar.radio(
    "Select Input Source",
    options=["Upload New Image", "Select from IAM Dataset"],
    index=0,
)

# Preprocessing pipeline options
pipeline_mode = st.sidebar.selectbox(
    "Image Preprocessing Pipeline",
    options=[
        ("enhanced_grayscale", "Enhanced Grayscale (Bilateral Filter + CLAHE)"),
        ("otsu_binarized", "Otsu Binarization (Global Threshold)"),
        ("adaptive_binarized", "Adaptive Gaussian Binarization"),
        ("raw", "Raw (No Preprocessing)"),
    ],
    format_func=lambda opt: opt[1],
    index=0,
)[0]

# Load sample dataset metadata
samples_metadata_path = os.path.join("dataset", "metadata.json")
samples_data = []
if os.path.exists(samples_metadata_path):
    with open(samples_metadata_path, "r", encoding="utf-8") as f:
        samples_data = json.load(f)

# Resolve target image and ground truth
target_image_bytes = None
ground_truth_text = None
source_name = "custom_upload"

if input_mode == "Upload New Image":
    uploaded_file = st.sidebar.file_uploader(
        "Upload handwritten note image",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
    )
    if uploaded_file is not None:
        target_image_bytes = uploaded_file.read()
        source_name = os.path.splitext(uploaded_file.name)[0]
else:
    if samples_data:
        sample_choice = st.sidebar.selectbox(
            "Choose IAM Handwriting Sample",
            options=samples_data,
            format_func=lambda s: f"{s['id']}: {s['ground_truth_text'][:35]}...",
        )
        if sample_choice:
            sample_img_path = os.path.join("dataset", "input_images", sample_choice["image_file"])
            if os.path.exists(sample_img_path):
                with open(sample_img_path, "rb") as f:
                    target_image_bytes = f.read()
                source_name = sample_choice["id"]
                ground_truth_text = sample_choice["ground_truth_text"]
    else:
        st.sidebar.warning("No sample images found in dataset folder.")

# Main Application Body
if target_image_bytes is not None:
    # Preprocess image
    with st.spinner("Preprocessing image..."):
        original_img, processed_img = preprocess_image(
            target_image_bytes, pipeline_mode=pipeline_mode
        )

    # Visual Inspection (Side by side)
    st.subheader("Image Preprocessing Comparison")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original Image**")
        st.image(original_img, channels="BGR", use_column_width=True)

    with col2:
        st.markdown(f"**Preprocessed Image ({pipeline_mode})**")
        # Grayscale images use single channel
        is_gray = len(processed_img.shape) == 2
        st.image(
            processed_img,
            channels="GRAY" if is_gray else "BGR",
            use_column_width=True,
        )

    # Extraction Trigger
    st.markdown("---")
    if st.button("Extract Handwritten Text", type="primary"):
        with st.spinner("Running EasyOCR inference..."):
            start_time = time.time()
            ocr_result = run_ocr(
                image=processed_img,
                save_output=True,
                base_name=source_name,
            )
            elapsed = round(time.time() - start_time, 2)

        st.subheader("Extracted Output")

        # Metrics display
        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            st.metric("Processing Time", f"{elapsed} s")
        with mcol2:
            st.metric("Confidence Score", f"{int(ocr_result['average_confidence'] * 100)} %")
        with mcol3:
            st.metric("Words Detected", ocr_result["word_count"])

        # Ground truth comparison if available
        if ground_truth_text:
            st.markdown("**Ground-Truth Reference (IAM Dataset):**")
            st.info(ground_truth_text)

        # Extracted Text Area
        st.markdown("**Extracted Text:**")
        extracted_text = st.text_area(
            label="Extracted Text (Editable)",
            value=ocr_result["text"],
            height=120,
            label_visibility="collapsed",
        )

        # Download Button and File Location
        col_btn, col_info = st.columns([1, 2])
        with col_btn:
            st.download_button(
                label="Download as Text File",
                data=extracted_text,
                file_name=f"extracted_{source_name}.txt",
                mime="text/plain",
            )
        with col_info:
            if ocr_result["saved_file"]:
                st.caption(f"Saved locally to: `{ocr_result['saved_file']}`")

else:
    st.info("Please upload an image from the sidebar or select a sample from the IAM dataset to begin.")
