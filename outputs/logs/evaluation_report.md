# OCR Accuracy Evaluation and Performance Report

**Project**: Handwritten Notes to Digital Text Converter  
**Dataset**: IAM Handwriting Database (7 Curated Line Samples)  
**OCR Engine**: EasyOCR (PyTorch CRAFT Detector + CRNN Recognizer)  
**Evaluation Date**: 2026-10-20  

---

## 1. Executive Summary

Optical character recognition on handwritten notes was quantitatively benchmarked against labeled ground-truth text from the IAM Handwriting Database. We evaluated four distinct computer vision preprocessing techniques:
1. **Raw Baseline**: Original image with no transformation.
2. **Otsu Binarization**: Global histogram-based thresholding.
3. **Adaptive Gaussian Binarization**: Local neighborhood thresholding.
4. **Enhanced Grayscale**: Bilateral edge-preserving filtering + Contrast Limited Adaptive Histogram Equalization (CLAHE).

### Benchmark Comparison

| Preprocessing Technique | Character Error Rate (CER) | Word Error Rate (WER) | Character Accuracy (%) |
| :--- | :---: | :---: | :---: |
| **Enhanced Grayscale (Bilateral + CLAHE)** | **0.4625** | **1.0367** | **53.75%** |
| **Otsu Binarization (Global Threshold)** | 0.4760 | 1.0377 | 52.40% |
| **Raw Baseline (No Preprocessing)** | 0.5008 | 1.1235 | 49.92% |
| **Adaptive Gaussian Binarization** | 0.5942 | 1.0633 | 40.58% |

---

## 2. Key Findings

1. **Enhanced Grayscale is the Top Performer**:
   Applying bilateral filtering followed by CLAHE yielded the highest character accuracy (**53.75%**) and lowest error rate (**0.4625 CER**), representing a **+3.83% absolute accuracy gain** over the raw baseline.
2. **Adaptive Thresholding Degrades Accuracy**:
   Adaptive Gaussian binarization performed the worst (**40.58% accuracy**). In handwriting, paper grain and ink bleed create micro-textures that adaptive filters mistake for edges, fragmenting cursive loops into broken dots.
3. **High Performance on Clear Writing**:
   On clear, printed-style handwriting (Sample 04: *"Lollobrigida . In the holiday seasonal months before and"*), Enhanced Grayscale achieved **68.52% character accuracy**.

---

## 3. Root Cause Analysis: Why Handwritten OCR Fails

Understanding why open-source models struggle with handwriting explains previous poor results:

### A. The Connected Stroke Problem (Ligatures)
In printed text, characters have clean white space between them (`c-a-s-t`). In cursive handwriting, ink flows continuously between letters without gaps. EasyOCR's CRNN recognizer relies on horizontal feature pooling; when characters are physically connected, the neural network cannot find boundary anchors, resulting in merged or substituted letters (e.g., `cast were well` becomes `Ibexl melll`).

### B. Visual Ambiguity in Slanted Letters
Handwritten letters with identical visual geometry frequently cause substitution errors:
- `a` vs. `o` vs. `u`
- `cl` vs. `d`
- `rn` vs. `m`
- `l` vs. `1` vs. `I`

### C. Dataset Training Domain Discrepancy
EasyOCR is trained primarily on scene text (road signs, billboards, storefronts, and printed documents). While it handles isolated, print-style handwriting relatively well, specialized handwriting recognition requires specialized Transformer models (like Microsoft TrOCR) fine-tuned specifically on IAM handwriting data.

---

## 4. Practical Recommendations for Users

To maximize OCR accuracy with this application:
1. **Always use the Enhanced Grayscale mode** (default in our app).
2. **Write in print or semi-print style** rather than heavy cursive.
3. **Capture photos in even, flat lighting** to prevent harsh shadow gradients.
4. **Keep the camera parallel** to the paper to prevent perspective distortion.
