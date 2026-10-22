---
title: Handwritten Api
emoji: 📝
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# Handwritten Notes to Digital Text Converter

An end-to-end Optical Character Recognition (OCR) web application and REST API that converts handwritten notes into digitized, editable text using OpenCV image preprocessing and EasyOCR deep learning models.

---

## 1. Project Overview

Handwritten notes are difficult to store, search, and edit. This project provides a complete, accessible system to digitize handwritten documents without requiring expensive commercial cloud APIs or training proprietary neural networks from scratch.

### Core Objectives
1. **Read handwritten text**: Ingest standard image formats (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`).
2. **Preprocess image**: Clean background paper grain and boost stroke contrast using OpenCV.
3. **Extract text using OCR**: Run open-source EasyOCR (CRAFT character detector + CRNN recognizer).
4. **Display and save output**: Render side-by-side previews, provide one-click clipboard copy, and save/download text files.

---

## 2. System Architecture

```
+-------------------------------------------------------------------------+
|                               FRONTEND                                  |
|                           (Hosted on Netlify)                           |
|                                                                         |
|   - Drag & Drop Image Dropzone             - Preprocessing Mode Select  |
|   - IAM Sample Dataset Gallery             - Side-by-Side Image Preview |
|   - Editable Text Area & Metrics           - Copy & Download (.txt)     |
+-------------------------------------------------------------------------+
                                    |
                HTTP POST /api/extract (Multipart Form-Data)
                                    |
                                    v
+-------------------------------------------------------------------------+
|                                BACKEND                                  |
|                     (Hosted on Hugging Face Spaces)                     |
|                                                                         |
|   FastAPI REST Service (CORS Enabled, Port 7860)                        |
|                                                                         |
|   1. OpenCV Computer Vision Preprocessing:                              |
|      - Grayscale Conversion                                             |
|      - Bilateral Filtering (Edge-preserving noise reduction)             |
|      - CLAHE (Contrast Limited Adaptive Histogram Equalization)         |
|      - Otsu & Adaptive Binarization (Optional Pipelines)                |
|                                                                         |
|   2. EasyOCR Deep Learning Inference:                                   |
|      - CRAFT Detector (Text region heatmaps & bounding boxes)           |
|      - Spatial Reading-Order Sorting (Line clustering & word sorting)   |
|      - CRNN Recognizer (ResNet + BiLSTM + CTC character decoding)       |
|                                                                         |
|   3. Persistence & Response Formatter:                                  |
|      - Saves extracted text to outputs/text_files/                      |
|      - Returns JSON with extracted text, confidence, & base64 previews  |
+-------------------------------------------------------------------------+
```

---

## 3. Directory Structure

```
Project_3_Handwritten-Text-Extraction/
│
├── dataset/
│   ├── input_images/          # Curated IAM handwriting sample images
│   ├── output_text/           # Ground-truth transcriptions
│   └── metadata.json          # Dataset catalog and sample descriptors
│
├── outputs/
│   ├── text_files/            # Automatically persisted extraction text files
│   └── logs/                  # Evaluation logs, metrics, and error traces
│
├── utils/
│   ├── __init__.py            # Python package initialization
│   ├── image_utils.py         # OpenCV image loading, resizing, CLAHE, denoising, binarization
│   └── ocr_utils.py           # EasyOCR singleton reader, reading-order sorting, text assembly
│
├── frontend/                  # Static Web Client (for Netlify)
│   ├── index.html             # Semantic interface structure
│   ├── styles.css             # Professional, high-contrast, responsive CSS
│   └── app.js                 # API controller, image preview, clipboard copy, file download
│
├── api.py                     # FastAPI REST API server (for Hugging Face Spaces)
├── app.py                     # Standalone Streamlit application (mandated by Project Specification)
├── Dockerfile                 # Container definition for Hugging Face Spaces
├── netlify.toml               # Deployment configuration for Netlify
├── requirements.txt           # Pinned Python package dependencies
└── run_app.bat                # Windows one-click local startup script
```

---

## 4. Local Installation and Setup

### Prerequisites
- Python 3.11 recommended (compatible with 3.10 – 3.12).
- Git.

### Step 1: Clone Repository and Navigate to Folder
```bash
git clone <your-repository-url>
cd Project_3_Handwritten-Text-Extraction
```

### Step 2: Create and Activate Virtual Environment
Using Conda:
```bash
conda create -n ocr_env python=3.11 -y
conda activate ocr_env
```
Or using standard Python `venv`:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. Running the Application Locally

### Option A: Run the FastAPI REST Server + Web Frontend (Recommended)
```bash
python -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser to: **`http://localhost:8000`**  
The full web client will open, ready to process images. Interactive Swagger API documentation is available at **`http://localhost:8000/docs`**.

### Option B: Run the Streamlit Desktop Application
```bash
streamlit run app.py
```
Open your browser to: **`http://localhost:8501`**.

---

## 6. Deployment Instructions

### 6.1 Deploy Backend to Hugging Face Spaces

1. Create a free account at [huggingface.co](https://huggingface.co).
2. Click **New Space** ([huggingface.co/new-space](https://huggingface.co/new-space)).
3. Space Settings:
   - **Space Name**: `handwritten-text-extractor`
   - **License**: `mit` or `apache-2.0`
   - **SDK**: Select **Docker** (Blank)
   - **Hardware**: Free CPU basic (2 vCPU, 16 GB RAM)
4. Push the project repository to Hugging Face:
   ```bash
   git remote add space https://huggingface.co/spaces/<YOUR-USERNAME>/handwritten-text-extractor
   git push space main
   ```
5. Hugging Face Spaces will automatically build the `Dockerfile`, cache the EasyOCR weights, and expose your API on port 7860.
6. Your backend API URL will be:
   `https://<YOUR-USERNAME>-handwritten-text-extractor.hf.space`

### 6.2 Deploy Frontend to Netlify

#### Method A: Netlify Drop (Instant Zero-CLI Deployment)
1. Log in to [netlify.com](https://www.netlify.com).
2. Go to **Sites** -> **Add new site** -> **Deploy manually**.
3. Drag and drop the `frontend` folder directly onto the upload area.
4. Netlify will publish the site immediately and provide a live URL (e.g., `https://handwritten-ocr.netlify.app`).

#### Method B: Git-Connected Netlify Deployment
1. Connect your GitHub repository to Netlify.
2. In the build settings:
   - **Base directory**: Leave blank or set to `.`
   - **Publish directory**: `frontend`
3. Click **Deploy Site**.

### 6.3 Connect Netlify Frontend to Hugging Face Backend
1. Open your live Netlify website in any browser.
2. In the top **Backend URL** field, enter your Hugging Face Space URL:
   `https://<YOUR-USERNAME>-handwritten-text-extractor.hf.space`
3. Click **Connect**. The status light will turn green once connected.

---

## 7. Deployment Considerations and Platform Behaviors

- **Hugging Face Sleeping Policy**: On the free tier, Hugging Face Spaces pauses containers after 48 hours of inactivity. When a new request arrives at a sleeping Space, it undergoes a cold start lasting approximately 30–45 seconds. The frontend features a connection status monitor and timeout handling to accommodate this.
- **Netlify CDN Reliability**: The Netlify frontend is statically served from global edge caches, meaning it never sleeps and loads instantaneously.
- **CORS Configuration**: The FastAPI backend includes `CORSMiddleware` configured with `allow_origins=["*"]`, enabling requests from any Netlify domain.

---

## 8. Empirical Accuracy Findings

Benchmarked against authentic ground-truth transcriptions from the IAM Handwriting Database:

| Pipeline Technique | Character Error Rate (CER) | Word Error Rate (WER) | Character Accuracy (%) |
| :--- | :---: | :---: | :---: |
| **Enhanced Grayscale (Bilateral + CLAHE)** | **0.4625** | **1.0367** | **53.75%** |
| **Otsu Binarization (Global Threshold)** | 0.4760 | 1.0377 | 52.40% |
| **Raw Baseline (No Preprocessing)** | 0.5008 | 1.1235 | 49.92% |
| **Adaptive Gaussian Binarization** | 0.5942 | 1.0633 | 40.58% |

**Key Takeaway**: Enhanced Grayscale provides the highest accuracy (+3.83% over baseline) by preserving continuous ink gradients and preventing the fragmentation caused by binarization filters.

---

## 9. License

This project is licensed under the MIT License. IAM Handwriting Database samples are used for educational and research evaluation under the University of Bern non-commercial evaluation terms.
