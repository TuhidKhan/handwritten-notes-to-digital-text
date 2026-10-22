"""Dual-Mode Application Entrypoint for Hugging Face Spaces (Gradio SDK) & Netlify.

This application provides:
1. Interactive Gradio Web Interface for Hugging Face Spaces.
2. Full FastAPI REST API endpoints (/health, /api/samples, /api/extract)
   with CORS enabled to power the Netlify web client.
"""

import os
import sys

# Compatibility shim for huggingface_hub and gradio oauth
try:
    import huggingface_hub
    if not hasattr(huggingface_hub, "HfFolder"):
        class _HfFolder:
            @classmethod
            def get_token(cls):
                return os.environ.get("HF_TOKEN")
            @classmethod
            def save_token(cls, token):
                pass
            @classmethod
            def delete_token(cls):
                pass
        huggingface_hub.HfFolder = _HfFolder
except Exception:
    pass

import uvicorn
import gradio as gr
from fastapi.responses import RedirectResponse

# Import FastAPI backend application and computer vision utilities
from api import app as fastapi_app
from utils.image_utils import preprocess_image
from utils.ocr_utils import run_ocr


@fastapi_app.get("/", include_in_schema=False)
def root_redirect():
    """Redirect root traffic to the interactive Gradio UI."""
    return RedirectResponse(url="/gradio")


def gradio_ocr_handler(image, pipeline_mode):
    """Processes image from the Gradio interface and returns extracted text."""
    if image is None:
        return "Please upload an image.", None
    original, processed = preprocess_image(image, pipeline_mode=pipeline_mode)
    result = run_ocr(processed, save_output=True, base_name="gradio_note")
    return result["text"], processed


# Create Gradio Blocks Interface
with gr.Blocks(title="Handwritten Notes to Digital Text Converter") as demo:
    gr.Markdown("# Handwritten Notes to Digital Text Converter")
    gr.Markdown("OpenCV Preprocessing & EasyOCR Extraction. Serving Netlify REST API.")

    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="numpy", label="Upload Handwritten Note")
            mode_select = gr.Dropdown(
                choices=[
                    ("Enhanced Grayscale (Bilateral + CLAHE)", "enhanced_grayscale"),
                    ("Otsu Binarization", "otsu_binarized"),
                    ("Adaptive Gaussian Binarization", "adaptive_binarized"),
                    ("Raw (No Preprocessing)", "raw"),
                ],
                value="enhanced_grayscale",
                label="Preprocessing Pipeline Mode",
            )
            extract_btn = gr.Button("Extract Handwritten Text", variant="primary")

        with gr.Column():
            proc_img_output = gr.Image(label="Preprocessed Image")
            text_output = gr.Textbox(label="Extracted Digital Text (Editable)", lines=6)

    extract_btn.click(
        fn=gradio_ocr_handler,
        inputs=[img_input, mode_select],
        outputs=[text_output, proc_img_output],
    )

# Mount Gradio into the FastAPI application
app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
