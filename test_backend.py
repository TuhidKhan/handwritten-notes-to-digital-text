"""Comprehensive Automated Backend and OCR Verification Suite.

This test script validates:
1. Image preprocessing transformations (grayscale, CLAHE, denoising, binarization).
2. OCR model inference, reading order sorting, and file persistence.
3. FastAPI REST endpoints (/health, /api/samples, /api/extract).
4. Edge cases: invalid file formats, empty files, missing inputs.
"""

import os
import sys
import json
import numpy as np
from fastapi.testclient import TestClient

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
from api import app

client = TestClient(app)


def print_test_header(title: str):
    print(f"\n{'='*70}\n[RUNNING TEST] {title}\n{'='*70}")


def test_image_utils():
    print_test_header("1. Image Preprocessing Utilities (utils/image_utils.py)")

    sample_path = os.path.join("dataset", "input_images", "sample_01.png")
    assert os.path.exists(sample_path), f"Sample image missing at {sample_path}"

    # Test 1.1: Loading from path
    bgr_img = load_image(sample_path)
    assert isinstance(bgr_img, np.ndarray), "Failed: Loaded image is not a numpy array"
    assert len(bgr_img.shape) == 3 and bgr_img.shape[2] == 3, "Failed: Expected 3-channel BGR image"
    print("  [PASS] load_image from file path: shape =", bgr_img.shape)

    # Test 1.2: Loading from raw bytes
    with open(sample_path, "rb") as f:
        raw_bytes = f.read()
    bgr_from_bytes = load_image(raw_bytes)
    assert bgr_from_bytes.shape == bgr_img.shape, "Failed: Byte loading produced mismatching dimensions"
    print("  [PASS] load_image from in-memory bytes")

    # Test 1.3: Grayscale conversion
    gray = to_grayscale(bgr_img)
    assert len(gray.shape) == 2, "Failed: Grayscale image must have 2 dimensions (H, W)"
    print("  [PASS] to_grayscale: shape =", gray.shape)

    # Test 1.4: CLAHE Contrast Adjustment
    enhanced = apply_clahe(gray)
    assert enhanced.shape == gray.shape, "Failed: CLAHE altered image dimensions"
    print("  [PASS] apply_clahe: dynamic range equalized")

    # Test 1.5: Noise reduction (bilateral and gaussian)
    denoised_bi = reduce_noise(gray, method="bilateral")
    denoised_gauss = reduce_noise(gray, method="gaussian")
    assert denoised_bi.shape == gray.shape and denoised_gauss.shape == gray.shape
    print("  [PASS] reduce_noise: bilateral and gaussian filtering verified")

    # Test 1.6: Binarization (Otsu and Adaptive)
    otsu = binarize(gray, method="otsu")
    adaptive = binarize(gray, method="adaptive")
    unique_otsu = set(np.unique(otsu))
    assert unique_otsu.issubset({0, 255}), f"Failed: Otsu image contains non-binary values: {unique_otsu}"
    print("  [PASS] binarize: Otsu and adaptive thresholding produce valid binary masks")

    # Test 1.7: Complete Preprocessing Pipeline
    orig, proc = preprocess_image(sample_path, pipeline_mode="enhanced_grayscale")
    assert orig is not None and proc is not None
    print("  [PASS] preprocess_image pipeline (enhanced_grayscale mode)")

    # Test 1.8: Base64 Data URL generation
    b64_url = image_to_base64(proc)
    assert b64_url.startswith("data:image/png;base64,"), "Failed: Invalid base64 data URL format"
    print("  [PASS] image_to_base64: valid Data URL created (length:", len(b64_url), ")")


def test_ocr_utils():
    print_test_header("2. OCR Engine and Postprocessing (utils/ocr_utils.py)")

    # Test 2.1: Reader singleton loading
    reader = get_ocr_reader()
    assert reader is not None, "Failed: EasyOCR reader failed to initialize"
    print("  [PASS] get_ocr_reader: singleton model loaded successfully")

    # Test 2.2: Spatial reading-order sorting
    # Synthetic out-of-order boxes representing 2 lines of text
    # Line 1: "Hello" (x=10, y=10) then "World" (x=60, y=10)
    # Line 2: "Testing" (x=10, y=50) then "OCR" (x=70, y=50)
    mock_detections = [
        ([[70, 50], [100, 50], [100, 70], [70, 70]], "OCR", 0.95),       # Line 2, Word 2
        ([[10, 50], [60, 50], [60, 70], [10, 70]], "Testing", 0.96),      # Line 2, Word 1
        ([[60, 10], [110, 10], [110, 30], [60, 30]], "World", 0.94),      # Line 1, Word 2
        ([[10, 10], [50, 10], [50, 30], [10, 30]], "Hello", 0.98),        # Line 1, Word 1
    ]
    sorted_res = sort_detections_in_reading_order(mock_detections)
    ordered_words = [item[1] for item in sorted_res]
    assert ordered_words == ["Hello", "World", "Testing", "OCR"], f"Failed sorting: Got {ordered_words}"
    print("  [PASS] sort_detections_in_reading_order: correctly recovered natural reading order")

    # Test 2.3: Text assembly and cleaning
    assembled = clean_and_assemble_text(sorted_res)
    assert assembled["text"] == "Hello World Testing OCR"
    assert assembled["word_count"] == 4
    assert assembled["average_confidence"] > 0.90
    print("  [PASS] clean_and_assemble_text: text merged, confidence calculated =", assembled["average_confidence"])

    # Test 2.4: Real inference on IAM dataset sample 01
    sample_path = os.path.join("dataset", "input_images", "sample_01.png")
    orig, processed = preprocess_image(sample_path, pipeline_mode="enhanced_grayscale")
    ocr_result = run_ocr(processed, reader=reader, save_output=True, base_name="test_sample_01")
    
    assert ocr_result["text"], "Failed: No text extracted from sample_01.png"
    assert ocr_result["saved_file"] and os.path.exists(ocr_result["saved_file"]), "Failed: Output file not persisted"
    print(f"  [PASS] run_ocr on sample_01: Extracted text: '{ocr_result['text']}'")
    print(f"         Confidence: {ocr_result['average_confidence']} | Saved file: {ocr_result['saved_file']}")


def test_api_endpoints():
    print_test_header("3. FastAPI REST Endpoints & Error Handling (api.py)")

    # Test 3.1: Root endpoint (serves index.html) and API info endpoint
    resp_root = client.get("/")
    assert resp_root.status_code == 200, f"Expected 200, got {resp_root.status_code}"
    assert "<!DOCTYPE html>" in resp_root.text
    print("  [PASS] GET / -> HTTP 200 OK (Serves frontend index.html)")

    resp_info = client.get("/api/info")
    assert resp_info.status_code == 200
    assert resp_info.json()["status"] == "online"
    print("  [PASS] GET /api/info -> HTTP 200 OK (API metadata)")

    # Test 3.2: Health check
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"
    print("  [PASS] GET /health -> HTTP 200 OK (Model Healthy)")

    # Test 3.3: Samples catalog
    resp = client.get("/api/samples")
    assert resp.status_code == 200
    samples = resp.json()["samples"]
    assert len(samples) == 7, f"Expected 7 IAM samples, got {len(samples)}"
    print(f"  [PASS] GET /api/samples -> HTTP 200 OK (Returned {len(samples)} curated IAM samples)")

    # Test 3.4: POST /api/extract with valid sample_id
    resp = client.post("/api/extract", data={"sample_id": "sample_01", "pipeline_mode": "enhanced_grayscale"})
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()
    assert data["success"] is True
    assert len(data["extracted_text"]) > 0
    assert "original_preview" in data and data["original_preview"].startswith("data:image/png;base64,")
    assert "preprocessed_preview" in data and data["preprocessed_preview"].startswith("data:image/png;base64,")
    print(f"  [PASS] POST /api/extract (sample_id='sample_01'): '{data['extracted_text']}'")

    # Test 3.5: POST /api/extract with multipart file upload
    sample_file_path = os.path.join("dataset", "input_images", "sample_02.png")
    with open(sample_file_path, "rb") as f:
        file_bytes = f.read()
    
    files = {"file": ("sample_02.png", file_bytes, "image/png")}
    resp = client.post("/api/extract", files=files, data={"pipeline_mode": "enhanced_grayscale"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    print(f"  [PASS] POST /api/extract (Multipart File Upload): '{data['extracted_text']}'")

    # Test 3.6: Error handling - Empty request (no file, no sample_id)
    resp = client.post("/api/extract", data={})
    assert resp.status_code == 400
    print("  [PASS] POST /api/extract (No input) -> HTTP 400 Bad Request correctly rejected")

    # Test 3.7: Error handling - Invalid file format (.txt instead of image)
    fake_txt = {"file": ("notes.txt", b"some raw text here", "text/plain")}
    resp = client.post("/api/extract", files=fake_txt)
    assert resp.status_code == 400
    assert "Unsupported file format" in resp.json()["detail"]
    print("  [PASS] POST /api/extract (Unsupported file extension) -> HTTP 400 correctly rejected")

    # Test 3.8: Error handling - Empty file (0 bytes)
    empty_file = {"file": ("empty.png", b"", "image/png")}
    resp = client.post("/api/extract", files=empty_file)
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()
    print("  [PASS] POST /api/extract (Empty file 0 bytes) -> HTTP 400 correctly rejected")

    # Test 3.9: Error handling - Non-existent sample_id
    resp = client.post("/api/extract", data={"sample_id": "non_existent_sample_99"})
    assert resp.status_code == 404
    print("  [PASS] POST /api/extract (Invalid sample_id) -> HTTP 404 correctly rejected")


if __name__ == "__main__":
    print(f"{'#'*70}\nSTARTING SUITE: BACKEND AND OCR PIPELINE VERIFICATION\n{'#'*70}")
    try:
        test_image_utils()
        test_ocr_utils()
        test_api_endpoints()
        print(f"\n{'#'*70}\nALL TESTS PASSED! BACKEND IS ROBUST, VERIFIED, AND PRODUCTION READY.\n{'#'*70}\n")
    except AssertionError as err:
        print(f"\nTEST ASSERTION FAILED: {err}")
        sys.exit(1)
    except Exception as exc:
        print(f"\nUNEXPECTED EXCEPTION DURING TESTING: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
