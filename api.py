"""FastAPI Backend Server for Handwritten Notes to Digital Text Converter.

This server provides REST API endpoints to process handwritten images, execute
computer vision preprocessing, run EasyOCR inference, and return structured JSON
with side-by-side preview images, confidence metrics, and download references.
"""

import os
import time
import json
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from utils.image_utils import (
    load_image,
    preprocess_image,
    image_to_base64,
)
from utils.ocr_utils import (
    get_ocr_reader,
    run_ocr,
)

# Initialize FastAPI application
app = FastAPI(
    title="Handwritten Notes to Digital Text Converter API",
    description="OpenCV + EasyOCR backend for digitizing handwritten text",
    version="1.0.0",
)

# Enable CORS (Cross-Origin Resource Sharing)
# This allows the Netlify frontend and local clients to communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins including Netlify domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supported image extensions and size limit (15 MB)
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024


@app.on_event("startup")
async def startup_event():
    """Pre-warms the EasyOCR neural network model into memory on server boot."""
    try:
        # Pre-load reader so the first user request does not incur cold-start latency
        get_ocr_reader()
    except Exception as e:
        print(f"Warning: OCR model preload failed on startup: {e}")


@app.get("/api/info")
def root_info():
    """API metadata and status endpoint."""
    return {
        "status": "online",
        "service": "Handwritten Notes to Digital Text Converter API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "samples": "/api/samples",
            "extract": "/api/extract (POST)",
        },
    }


@app.get("/health")
def health_check():
    """Health check endpoint for deployment monitoring and availability probes."""
    return {
        "status": "healthy",
        "model": "EasyOCR (CRAFT + CRNN)",
        "framework": "FastAPI + PyTorch",
    }


@app.get("/api/samples")
def list_sample_images():
    """Returns curated IAM handwriting dataset samples with ground-truth metadata.

    Allows the frontend to offer pre-loaded test images for instant evaluation.
    """
    metadata_path = os.path.join("dataset", "metadata.json")
    if not os.path.exists(metadata_path):
        return {"samples": []}

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        enriched_samples = []
        for s in samples:
            img_path = os.path.join("dataset", "input_images", s["image_file"])
            if os.path.exists(img_path):
                img = load_image(img_path)
                preview_b64 = image_to_base64(img)
                enriched_samples.append(
                    {
                        "id": s["id"],
                        "filename": s["image_file"],
                        "ground_truth": s["ground_truth_text"],
                        "source": s["dataset_source"],
                        "preview": preview_b64,
                    }
                )
        return {"samples": enriched_samples}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load sample dataset: {str(e)}",
        )


@app.post("/api/extract")
async def extract_text_endpoint(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    pipeline_mode: str = Form("enhanced_grayscale"),
):
    """Processes an image and extracts handwritten text.

    Args:
        file: Uploaded image file (multipart/form-data).
        sample_id: Optional ID of a built-in sample (e.g. 'sample_01').
        pipeline_mode: Preprocessing technique:
            - 'enhanced_grayscale' (Default: Grayscale + Bilateral Filter + CLAHE)
            - 'otsu_binarized' (Otsu's global binarization)
            - 'adaptive_binarized' (Adaptive Gaussian thresholding)
            - 'raw' (Original image)

    Returns:
        JSON response with extracted text, confidence, side-by-side previews,
        processing metrics, and output file path.
    """
    start_time = time.time()
    image_bytes = None
    source_name = "uploaded_note"

    # Step 1: Input resolution (Uploaded file vs built-in sample)
    if file is not None and file.filename:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format '{ext}'. Supported formats: {', '.join(ALLOWED_EXTENSIONS)}",
            )

        image_bytes = await file.read()
        source_name = os.path.splitext(file.filename)[0]

        if len(image_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty (0 bytes).",
            )

        if len(image_bytes) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES // (1024*1024)} MB.",
            )

    elif sample_id:
        # Load directly from dataset/input_images
        sample_filename = f"{sample_id}.png"
        sample_path = os.path.join("dataset", "input_images", sample_filename)
        if not os.path.exists(sample_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sample '{sample_id}' not found.",
            )
        with open(sample_path, "rb") as f:
            image_bytes = f.read()
        source_name = sample_id

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image provided. Please upload an image file or specify a sample_id.",
        )

    # Step 2: Computer Vision Preprocessing
    try:
        original_img, processed_img = preprocess_image(
            image_bytes, pipeline_mode=pipeline_mode
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Image decoding and preprocessing failed: {str(e)}",
        )

    # Step 3: OCR Inference and Text Reconstruction
    try:
        ocr_result = run_ocr(
            image=processed_img,
            save_output=True,
            base_name=source_name,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR inference engine failed: {str(e)}",
        )

    # Step 4: Generate Base64 previews for client side-by-side rendering
    orig_preview = image_to_base64(original_img)
    proc_preview = image_to_base64(processed_img)

    elapsed_time = round(time.time() - start_time, 3)

    return {
        "success": True,
        "extracted_text": ocr_result["text"],
        "average_confidence": ocr_result["average_confidence"],
        "word_count": ocr_result["word_count"],
        "processing_time_sec": elapsed_time,
        "pipeline_mode": pipeline_mode,
        "saved_file": ocr_result["saved_file"],
        "details": ocr_result["details"],
        "original_preview": orig_preview,
        "preprocessed_preview": proc_preview,
    }


# Mount static frontend files for unified single-server deployment
frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    # Hugging Face Spaces binds to port 7860 by default; local default is 8000
    port = int(os.getenv("PORT", 7860 if os.getenv("SPACE_ID") else 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)
