"""OCR Model Inference and Text Postprocessing Utilities.

This module encapsulates EasyOCR initialization, text detection, character
recognition, spatial reading-order sorting, text cleaning, and disk persistence.
"""

import os
import re
import datetime
from typing import List, Dict, Tuple, Any, Optional
import easyocr
import numpy as np
import torch

# Global singleton cache for the EasyOCR Reader to avoid re-allocating model weights
_CACHED_READER: Optional[easyocr.Reader] = None


def get_ocr_reader(
    languages: Optional[List[str]] = None,
    gpu: Optional[bool] = None,
    download_enabled: bool = True,
) -> easyocr.Reader:
    """Retrieves or initializes a cached singleton instance of the EasyOCR Reader.

    Args:
        languages: List of language codes (defaults to ['en']).
        gpu: Boolean flag to enable GPU acceleration. If None, auto-detects CUDA.
        download_enabled: Whether to download model weights if missing.

    Returns:
        easyocr.Reader: Ready-to-use OCR inference engine.
    """
    global _CACHED_READER

    if _CACHED_READER is not None:
        return _CACHED_READER

    if languages is None:
        languages = ["en"]

    if gpu is None:
        gpu = torch.cuda.is_available()

    # Initialize EasyOCR reader (loads CRAFT detector and CRNN recognizer into memory)
    _CACHED_READER = easyocr.Reader(
        lang_list=languages,
        gpu=gpu,
        download_enabled=download_enabled,
        verbose=False,
    )
    return _CACHED_READER


def sort_detections_in_reading_order(
    detections: List[Tuple[List[List[int]], str, float]],
    y_threshold_ratio: float = 0.5,
) -> List[Tuple[List[List[int]], str, float]]:
    """Sorts OCR bounding boxes into natural reading order: top-to-bottom, left-to-right.

    Because handwriting strokes can be detected out of linear order by the CRAFT model,
    we group detections into horizontal text lines using vertical centroid proximity,
    sort lines by vertical position, and sort words within each line by horizontal X position.

    Args:
        detections: List of tuples (bbox, text, confidence) from EasyOCR.
        y_threshold_ratio: Multiplier of line height to determine line clustering.

    Returns:
        List[Tuple[List[List[int]], str, float]]: Sorted list of detections.
    """
    if not detections:
        return []

    # Calculate bounding box metrics for each detection
    items = []
    for bbox, text, conf in detections:
        # bbox is [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        y_center = (y_min + y_max) / 2.0
        height = max(1, y_max - y_min)

        items.append(
            {
                "bbox": bbox,
                "text": text.strip(),
                "conf": float(conf),
                "x_min": x_min,
                "y_min": y_min,
                "y_center": y_center,
                "height": height,
            }
        )

    # Sort initially by top Y coordinate
    items.sort(key=lambda item: item["y_center"])

    # Cluster detections into distinct horizontal lines
    lines = []
    current_line = []
    current_line_y = None
    avg_line_height = np.median([item["height"] for item in items]) if items else 20.0
    line_threshold = avg_line_height * y_threshold_ratio

    for item in items:
        if current_line_y is None:
            current_line.append(item)
            current_line_y = item["y_center"]
        else:
            if abs(item["y_center"] - current_line_y) <= line_threshold:
                current_line.append(item)
                # Update running average Y for this line
                current_line_y = sum(i["y_center"] for i in current_line) / len(current_line)
            else:
                # Line completed; sort words in line from left to right
                current_line.sort(key=lambda i: i["x_min"])
                lines.append(current_line)
                current_line = [item]
                current_line_y = item["y_center"]

    if current_line:
        current_line.sort(key=lambda i: i["x_min"])
        lines.append(current_line)

    # Flatten the lines back to a single sorted list
    sorted_detections = []
    for line in lines:
        for item in line:
            sorted_detections.append((item["bbox"], item["text"], item["conf"]))

    return sorted_detections


def clean_and_assemble_text(
    sorted_detections: List[Tuple[List[List[int]], str, float]],
    confidence_threshold: float = 0.10,
    preserve_newlines: bool = True,
) -> Dict[str, Any]:
    """Cleans raw text segments and reconstructs readable paragraphs with confidence scoring.

    Args:
        sorted_detections: Sorted list of (bbox, text, confidence).
        confidence_threshold: Minimum confidence score to include a detected segment.
        preserve_newlines: Whether to preserve separate lines using newlines.

    Returns:
        Dict[str, Any]: {
            'text': str,
            'lines': List[str],
            'average_confidence': float,
            'word_count': int,
            'details': List[Dict]
        }
    """
    if not sorted_detections:
        return {
            "text": "",
            "lines": [],
            "average_confidence": 0.0,
            "word_count": 0,
            "details": [],
        }

    items = []
    confidences = []
    details = []

    for bbox, raw_text, conf in sorted_detections:
        # Strip excessive whitespace and unwanted non-printable control characters
        cleaned = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", raw_text).strip()
        if not cleaned:
            continue

        if conf >= confidence_threshold:
            xs = [pt[0] for pt in bbox]
            ys = [pt[1] for pt in bbox]
            items.append(
                {
                    "text": cleaned,
                    "conf": float(conf),
                    "bbox": [[int(pt[0]), int(pt[1])] for pt in bbox],
                    "y_center": (min(ys) + max(ys)) / 2.0,
                    "height": max(1, max(ys) - min(ys)),
                }
            )
            confidences.append(float(conf))
            details.append(
                {
                    "text": cleaned,
                    "confidence": round(float(conf), 4),
                    "bbox": [[int(pt[0]), int(pt[1])] for pt in bbox],
                }
            )

    if not items:
        return {
            "text": "",
            "lines": [],
            "average_confidence": 0.0,
            "word_count": 0,
            "details": [],
        }

    # Group into distinct lines if preserve_newlines is requested
    if preserve_newlines and len(items) > 1:
        avg_height = float(np.median([it["height"] for it in items]))
        line_thresh = avg_height * 0.6
        lines = []
        current_words = []
        current_y = None

        for it in items:
            if current_y is None:
                current_words.append(it["text"])
                current_y = it["y_center"]
            elif abs(it["y_center"] - current_y) <= line_thresh:
                current_words.append(it["text"])
            else:
                lines.append(" ".join(current_words))
                current_words = [it["text"]]
                current_y = it["y_center"]

        if current_words:
            lines.append(" ".join(current_words))
        full_text = "\n".join(lines)
    else:
        full_text = " ".join([it["text"] for it in items])
        lines = [full_text] if full_text else []

    avg_conf = float(np.mean(confidences)) if confidences else 0.0

    return {
        "text": full_text,
        "lines": lines,
        "average_confidence": round(avg_conf, 4),
        "word_count": len(items),
        "details": details,
    }


def save_text_to_disk(
    text: str,
    output_dir: str = "outputs/text_files",
    base_name: str = "note",
) -> str:
    """Saves the extracted text into a timestamped .txt file inside the outputs folder.

    Args:
        text: Extracted text to persist.
        output_dir: Target directory path.
        base_name: Identifier prefix for the saved file.

    Returns:
        str: Absolute or normalized file path of the saved text file.
    """
    os.makedirs(output_dir, exist_ok=True)
    clean_base = re.sub(r"[^\w\-_\.]", "_", os.path.splitext(base_name)[0])
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"extracted_{clean_base}_{timestamp}.txt"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)

    return filepath


def run_ocr(
    image: np.ndarray,
    reader: Optional[easyocr.Reader] = None,
    languages: Optional[List[str]] = None,
    save_output: bool = True,
    base_name: str = "note",
) -> Dict[str, Any]:
    """High-level function executing EasyOCR inference, sorting, cleaning, and file saving.

    Args:
        image: Preprocessed OpenCV image matrix (grayscale or BGR).
        reader: EasyOCR reader instance (if None, uses singleton).
        languages: Language list if initializing reader dynamically.
        save_output: Whether to automatically save extracted text to disk.
        base_name: Prefix name for the saved output file.

    Returns:
        Dict[str, Any]: Comprehensive result dictionary including extracted text,
                        confidence, word details, and saved file path.
    """
    if reader is None:
        reader = get_ocr_reader(languages=languages or ["en"])

    # Step 1: Run deep learning inference
    raw_results = reader.readtext(image, detail=1, paragraph=False)

    # Step 2: Sort detections into natural reading order
    sorted_results = sort_detections_in_reading_order(raw_results)

    # Step 3: Clean and assemble text
    assembled = clean_and_assemble_text(sorted_results)

    # Step 4: Persist to disk if requested and non-empty
    saved_filepath = None
    if save_output and assembled["text"]:
        saved_filepath = save_text_to_disk(assembled["text"], base_name=base_name)

    assembled["saved_file"] = saved_filepath
    return assembled
