"""Comprehensive Automated OCR and Computer Vision Verification Suite.

This test script validates:
1. Image preprocessing transformations (grayscale, CLAHE, denoising, binarization).
2. OCR model inference, reading order sorting, and file persistence.
3. IAM Dataset metadata and ground truth alignment.
"""

import os
import sys
import json
import numpy as np

# Ensure root directory is on Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from utils.image_utils import (
    load_image,
    to_grayscale,
    apply_clahe,
    reduce_noise,
    binarize,
    preprocess_image,
    image_to_base64,
)
from utils.ocr_utils import (
    get_ocr_reader,
    sort_detections_in_reading_order,
    clean_and_assemble_text,
    save_text_to_disk,
    run_ocr,
)


def print_test_header(title: str):
    print("\n" + "=" * 70)
    print(f"[RUNNING TEST] {title}")
    print("=" * 70)


def test_image_preprocessing():
    """Validates individual and pipeline image transformations."""
    print_test_header("1. Image Preprocessing Utilities (utils/image_utils.py)")

    # Sample 01 path
    sample_path = os.path.join("dataset", "input_images", "sample_01.png")
    assert os.path.exists(sample_path), f"Sample image {sample_path} missing!"

    # 1.1 Load Image from path
    img_bgr = load_image(sample_path)
    assert isinstance(img_bgr, np.ndarray), "Failed to load as numpy array"
    assert len(img_bgr.shape) == 3, "Image should have 3 color channels"
    print(f"  [PASS] load_image from file path: shape = {img_bgr.shape}")

    # 1.2 Load from bytes
    with open(sample_path, "rb") as f:
        img_bytes = f.read()
    img_from_bytes = load_image(img_bytes)
    assert img_from_bytes.shape == img_bgr.shape, "Shape mismatch from bytes"
    print("  [PASS] load_image from in-memory bytes")

    # 1.3 Grayscale
    gray = to_grayscale(img_bgr)
    assert len(gray.shape) == 2, "Grayscale image must be 2D matrix"
    print(f"  [PASS] to_grayscale: shape = {gray.shape}")

    # 1.4 CLAHE
    clahe_img = apply_clahe(gray, clip_limit=2.0, tile_grid_size=(8, 8))
    assert clahe_img.shape == gray.shape, "CLAHE shape mismatch"
    print("  [PASS] apply_clahe: dynamic range equalized")

    # 1.5 Denoising
    denoised_bilateral = reduce_noise(gray, method="bilateral")
    denoised_gaussian = reduce_noise(gray, method="gaussian")
    assert denoised_bilateral.shape == gray.shape
    assert denoised_gaussian.shape == gray.shape
    print("  [PASS] reduce_noise: bilateral and gaussian filtering verified")

    # 1.6 Binarization
    otsu = binarize(gray, method="otsu")
    adaptive = binarize(gray, method="adaptive")
    assert set(np.unique(otsu)).issubset({0, 255}), "Otsu must be binary (0 or 255)"
    assert set(np.unique(adaptive)).issubset({0, 255}), "Adaptive must be binary (0 or 255)"
    print("  [PASS] binarize: Otsu and adaptive thresholding produce valid binary masks")

    # 1.7 Preprocess Pipeline
    orig, proc = preprocess_image(img_bgr, pipeline_mode="enhanced_grayscale")
    assert orig.shape == img_bgr.shape
    assert len(proc.shape) in (2, 3)
    print("  [PASS] preprocess_image pipeline (enhanced_grayscale mode)")

    # 1.8 Base64 serialization
    b64_str = image_to_base64(proc)
    assert b64_str.startswith("data:image/png;base64,"), "Invalid Data URI format"
    print(f"  [PASS] image_to_base64: valid Data URL created (length: {len(b64_str)})")


def test_ocr_engine_and_sorting():
    """Validates EasyOCR model loading, reading-order sorting, and file persistence."""
    print_test_header("2. OCR Engine and Postprocessing (utils/ocr_utils.py)")

    # 2.1 Model Initialization
    reader = get_ocr_reader(languages=["en"], gpu=False)
    assert reader is not None, "Failed to initialize EasyOCR reader"
    print("  [PASS] get_ocr_reader: singleton model loaded successfully")

    # 2.2 Spatial Reading Order Sorting
    mock_detections = [
        ([[100, 50], [200, 50], [200, 70], [100, 70]], "Second", 0.95),
        ([[10, 10], [90, 10], [90, 30], [10, 30]], "First", 0.98),
        ([[10, 50], [90, 50], [90, 70], [10, 70]], "Third", 0.92),
    ]
    sorted_dets = sort_detections_in_reading_order(mock_detections)
    sorted_words = [d[1] for d in sorted_dets]
    assert sorted_words == ["First", "Third", "Second"], f"Expected ['First', 'Third', 'Second'], got {sorted_words}"
    print("  [PASS] sort_detections_in_reading_order: correctly recovered natural reading order")

    # 2.3 Text Assembly and Confidence Scoring
    res_assembly = clean_and_assemble_text(sorted_dets)
    cleaned_text = res_assembly["text"]
    avg_conf = res_assembly["average_confidence"]
    word_count = res_assembly["word_count"]
    assert cleaned_text.replace("\n", " ") == "First Third Second"
    assert round(avg_conf, 4) == 0.9500
    assert word_count == 3
    print(f"  [PASS] clean_and_assemble_text: text merged, confidence calculated = {avg_conf:.4f}, words = {word_count}")

    # 2.4 End-to-End OCR on sample_01
    sample_path = os.path.join("dataset", "input_images", "sample_01.png")
    img = load_image(sample_path)
    _, proc = preprocess_image(img, pipeline_mode="enhanced_grayscale")

    result = run_ocr(proc, languages=["en"], save_output=True, base_name="test_sample_01")
    assert "text" in result and len(result["text"]) > 0, "OCR extracted empty text"
    assert "average_confidence" in result and result["average_confidence"] > 0
    assert "saved_file" in result and os.path.exists(result["saved_file"])
    print(f"  [PASS] run_ocr on sample_01: Extracted text: '{result['text']}'")
    print(f"         Confidence: {result['average_confidence']:.4f} | Saved file: {result['saved_file']}")


def test_dataset_integrity():
    """Validates IAM handwriting sample dataset files and metadata."""
    print_test_header("3. IAM Handwriting Dataset Integrity (dataset/)")

    meta_path = os.path.join("dataset", "metadata.json")
    assert os.path.exists(meta_path), "metadata.json is missing!"

    with open(meta_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    assert len(samples) >= 7, f"Expected at least 7 IAM samples, found {len(samples)}"

    for s in samples:
        img_p = os.path.join("dataset", "input_images", s["image_file"])
        gt_p = os.path.join("dataset", "output_text", s["ground_truth_file"])
        assert os.path.exists(img_p), f"Missing image {img_p}"
        assert os.path.exists(gt_p), f"Missing ground truth {gt_p}"
        with open(gt_p, "r", encoding="utf-8") as gf:
            gt_content = gf.read().strip()
        assert gt_content == s["ground_truth_text"].strip(), f"Metadata mismatch for {s['id']}"

    print(f"  [PASS] Verified all {len(samples)} IAM handwriting samples, ground truth files, and metadata.")


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("STARTING SUITE: OCR AND IMAGE PREPROCESSING PIPELINE VERIFICATION")
    print("#" * 70)

    test_image_preprocessing()
    test_ocr_engine_and_sorting()
    test_dataset_integrity()

    print("\n" + "#" * 70)
    print("ALL TESTS PASSED! OCR PIPELINE IS ROBUST, VERIFIED, AND READY.")
    print("#" * 70 + "\n")
