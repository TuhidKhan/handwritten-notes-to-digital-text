# Handwritten Notes to Digital Text Converter

An end-to-end Optical Character Recognition (OCR) web application that converts handwritten notes into digitized, editable text using OpenCV image preprocessing and EasyOCR deep learning models.

---

## 1. Project Overview

Handwritten notes are difficult to store, search, and edit. This application provides a streamlined, accessible interface to digitize handwritten documents without requiring expensive commercial cloud APIs.

### Core Objectives
1. **Read handwritten text**: Ingest standard image formats (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`).
2. **Preprocess image**: Clean background paper grain and boost stroke contrast using OpenCV.
3. **Extract text using OCR**: Run open-source EasyOCR (CRAFT character detector + CRNN recognizer).
4. **Display and save output**: Render side-by-side previews, provide one-click `.txt` downloads, and persist extractions to disk.

---

## 2. Directory Structure

```
Project_3_Handwritten-Text-Extraction/
|
|-- dataset/
|   |-- input_images/          # Curated IAM handwriting sample images
|   |-- output_text/           # Ground-truth transcriptions
|   `-- metadata.json          # Dataset catalog and sample descriptors
|
|-- outputs/
|   |-- text_files/            # Automatically persisted extraction text files
|   `-- logs/                  # Evaluation logs, metrics, and error traces
|
|-- utils/
|   |-- __init__.py            # Python package initialization
|   |-- image_utils.py         # OpenCV image loading, resizing, CLAHE, denoising, binarization
|   `-- ocr_utils.py           # EasyOCR reader, reading-order sorting, text assembly
|
|-- app.py                     # Main Streamlit application
|-- evaluate_accuracy.py       # Quantitative benchmark suite (CER, WER, accuracy)
|-- test_backend.py            # Unit test suite verifying CV pipeline and OCR functions
|-- requirements.txt           # Pinned Python dependencies
|-- run_app.bat                # Windows 1-click startup script
`-- README.md                  # System documentation
```

---

## 3. Local Installation and Setup

### Prerequisites
- Python 3.10 to 3.12 (Python 3.11 recommended).
- Git.

### Step 1: Clone Repository
```bash
git clone https://github.com/TuhidKhan/handwritten-notes-to-digital-text.git
cd handwritten-notes-to-digital-text
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
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 4. Running the Application Locally

Run the Streamlit application:
```bash
streamlit run app.py
```
Or on Windows, simply double-click `run_app.bat`.

The application will launch in your browser at `http://localhost:8501`.

---

## 5. Deploying to Streamlit Community Cloud

Streamlit Community Cloud provides 100% free hosting directly integrated with your GitHub repository:

1. Visit [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **Create app** (or **New app**).
3. Fill in the repository details:
   - **Repository**: `TuhidKhan/handwritten-notes-to-digital-text`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Click **Deploy!**.
5. Streamlit will install the requirements from `requirements.txt` and serve the application live.

---

## 6. Empirical Accuracy Evaluation

Benchmarked against authentic ground-truth transcriptions from the IAM Handwriting Database:

| Pipeline Technique | Character Error Rate (CER) | Word Error Rate (WER) | Character Accuracy (%) |
| :--- | :---: | :---: | :---: |
| Enhanced Grayscale (Bilateral + CLAHE) | 0.4625 | 1.0367 | 53.75% |
| Otsu Binarization (Global Threshold) | 0.4760 | 1.0377 | 52.40% |
| Raw Baseline (No Preprocessing) | 0.5008 | 1.1235 | 49.92% |
| Adaptive Gaussian Binarization | 0.5942 | 1.0633 | 40.58% |

Key takeaway: Enhanced Grayscale provides the highest accuracy (+3.83% over baseline) by preserving continuous ink gradients and preventing the fragmentation caused by binarization filters.

---

## 7. License

This project is licensed under the MIT License. IAM Handwriting Database samples are used for educational and research evaluation under the University of Bern non-commercial evaluation terms.
