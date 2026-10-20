"""End-to-End Workflow Integration Test.

Simulates the complete user interaction flow:
1. Browser loading the frontend web application (HTML, CSS, JS).
2. Connecting to the backend /health probe.
3. Loading IAM sample dataset catalog.
4. Extracting text across multiple preprocessing pipelines.
5. Verifying that disk files are saved in outputs/text_files/.
"""

import os
import sys
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from api import app

client = TestClient(app)


def test_e2e_flow():
    print("\n" + "=" * 70)
    print("STAGE 8: END-TO-END WORKFLOW INTEGRATION TEST")
    print("=" * 70)

    # Step 1: Verify Frontend Static Assets are Served
    print("\n[Step 1] Testing Frontend Serving via FastAPI...")
    resp = client.get("/")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    assert "Handwritten Notes to Digital Text Converter" in resp.text
    print("  [PASS] index.html successfully served at / (Length:", len(resp.text), "bytes)")

    resp_css = client.get("/styles.css")
    assert resp_css.status_code == 200
    assert "--color-primary" in resp_css.text
    print("  [PASS] styles.css successfully served (Length:", len(resp_css.text), "bytes)")

    resp_js = client.get("/app.js")
    assert resp_js.status_code == 200
    assert "checkBackendHealth" in resp_js.text
    print("  [PASS] app.js successfully served (Length:", len(resp_js.text), "bytes)")

    # Step 2: Verify Health Check Connection
    print("\n[Step 2] Testing Client-to-Backend Health Handshake...")
    resp_health = client.get("/health")
    assert resp_health.status_code == 200
    health_data = resp_health.json()
    assert health_data["status"] == "healthy"
    print("  [PASS] /health handshake successful:", health_data)

    # Step 3: Fetch Sample Dataset Catalog
    print("\n[Step 3] Fetching IAM Dataset Samples for UI Gallery...")
    resp_samples = client.get("/api/samples")
    assert resp_samples.status_code == 200
    samples = resp_samples.json()["samples"]
    assert len(samples) >= 5, f"Expected at least 5 samples, found {len(samples)}"
    print(f"  [PASS] Retrieved {len(samples)} curated IAM samples with thumbnails.")

    # Step 4: Simulate Extracting All 4 Preprocessing Modes on sample_02
    print("\n[Step 4] Testing All 4 Preprocessing Pipelines on Sample 02...")
    pipelines = ["enhanced_grayscale", "otsu_binarized", "adaptive_binarized", "raw"]
    
    for mode in pipelines:
        resp_extract = client.post(
            "/api/extract",
            data={"sample_id": "sample_02", "pipeline_mode": mode},
        )
        assert resp_extract.status_code == 200, f"Mode {mode} failed: {resp_extract.text}"
        data = resp_extract.json()
        assert data["success"] is True
        assert "preprocessed_preview" in data
        assert "original_preview" in data
        assert data["saved_file"] and os.path.exists(data["saved_file"])
        print(f"  [PASS] Mode: {mode:<20} | Time: {data['processing_time_sec']}s | Conf: {data['average_confidence']:.2f} | Text: '{data['extracted_text'][:40]}...'")

    # Step 5: Simulate User File Upload of a Note Image
    print("\n[Step 5] Testing Multipart File Upload (simulating file drag-and-drop)...")
    sample_file_path = os.path.join("dataset", "input_images", "sample_03.png")
    with open(sample_file_path, "rb") as f:
        img_bytes = f.read()

    files = {"file": ("my_handwritten_note.png", img_bytes, "image/png")}
    resp_upload = client.post(
        "/api/extract",
        files=files,
        data={"pipeline_mode": "enhanced_grayscale"},
    )
    assert resp_upload.status_code == 200
    upload_data = resp_upload.json()
    assert upload_data["success"] is True
    assert upload_data["word_count"] > 0
    print(f"  [PASS] Upload processed in {upload_data['processing_time_sec']}s")
    print(f"         Extracted: '{upload_data['extracted_text']}'")
    print(f"         Saved file: {upload_data['saved_file']}")

    print("\n" + "=" * 70)
    print("STAGE 8 INTEGRATION COMPLETE: ALL SYSTEMS VERIFIED AND OPERATIONAL")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        test_e2e_flow()
    except Exception as e:
        print(f"\n[FAIL] E2E Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
