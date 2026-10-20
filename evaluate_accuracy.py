"""Quantitative Accuracy Evaluation and Preprocessing Optimization Suite.

This script benchmarks EasyOCR on the curated IAM Handwriting Dataset.
Metrics computed:
- Character Error Rate (CER) via Levenshtein Edit Distance.
- Word Error Rate (WER).
- Average Confidence Score.
- Comparison across:
  * Raw baseline
  * Otsu binarization
  * Adaptive binarization
  * Enhanced Grayscale (CLAHE + Bilateral)
  * Enhanced Grayscale + Slanted/Cursive Model Tuning
"""

import os
import sys
import json
import re
from typing import List, Tuple, Dict, Any

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from utils.image_utils import preprocess_image, load_image
from utils.ocr_utils import (
    get_ocr_reader,
    sort_detections_in_reading_order,
    clean_and_assemble_text,
)


def levenshtein_distance(seq1: List[Any], seq2: List[Any]) -> int:
    """Calculates Levenshtein edit distance (insertions, deletions, substitutions)."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # Deletion
                dp[i][j - 1] + 1,       # Insertion
                dp[i - 1][j - 1] + cost # Substitution
            )

    return dp[m][n]


def normalize_text(text: str) -> str:
    """Normalizes whitespace and removes punctuation for fair word/char comparisons."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_metrics(ground_truth: str, prediction: str) -> Dict[str, float]:
    """Computes CER (Character Error Rate) and WER (Word Error Rate)."""
    gt_norm = normalize_text(ground_truth)
    pred_norm = normalize_text(prediction)

    gt_chars = list(gt_norm)
    pred_chars = list(pred_norm)

    # Character Error Rate
    char_distance = levenshtein_distance(gt_chars, pred_chars)
    total_chars = max(1, len(gt_chars))
    cer = char_distance / total_chars

    # Word Error Rate
    gt_words = gt_norm.split()
    pred_words = pred_norm.split()
    word_distance = levenshtein_distance(gt_words, pred_words)
    total_words = max(1, len(gt_words))
    wer = word_distance / total_words

    # Accuracy percentage (bounded between 0% and 100%)
    char_acc = max(0.0, 1.0 - cer) * 100.0

    return {
        "cer": round(cer, 4),
        "wer": round(wer, 4),
        "char_accuracy_pct": round(char_acc, 2),
        "char_distance": char_distance,
        "word_distance": word_distance,
        "gt_length": len(gt_chars),
        "pred_length": len(pred_chars),
    }


def run_benchmark():
    print("=" * 75)
    print("STAGE 9: OCR ACCURACY AND PREPROCESSING BENCHMARK")
    print("=" * 75)

    metadata_file = os.path.join("dataset", "metadata.json")
    with open(metadata_file, "r", encoding="utf-8") as f:
        samples = json.load(f)

    reader = get_ocr_reader()

    # Preprocessing pipelines to evaluate
    pipelines = [
        ("raw", "Raw Baseline (No CV Preprocessing)"),
        ("otsu_binarized", "Otsu Binarization (Global Threshold)"),
        ("adaptive_binarized", "Adaptive Gaussian Binarization"),
        ("enhanced_grayscale", "Enhanced Grayscale (Bilateral + CLAHE)"),
    ]

    pipeline_summaries = {}

    for mode_key, mode_name in pipelines:
        print(f"\nEvaluating Pipeline: {mode_name}")
        print("-" * 75)

        total_cer = 0.0
        total_wer = 0.0
        total_acc = 0.0
        sample_results = []

        for sample in samples:
            img_path = os.path.join("dataset", "input_images", sample["image_file"])
            gt_text = sample["ground_truth_text"]

            # Step 1: Preprocess
            _, processed_img = preprocess_image(img_path, pipeline_mode=mode_key)

            # Step 2: Inference with default EasyOCR settings
            raw_detections = reader.readtext(processed_img, detail=1, paragraph=False)
            sorted_detections = sort_detections_in_reading_order(raw_detections)
            extracted = clean_and_assemble_text(sorted_detections)
            pred_text = extracted["text"]

            # Step 3: Compute CER and WER
            metrics = compute_metrics(gt_text, pred_text)
            total_cer += metrics["cer"]
            total_wer += metrics["wer"]
            total_acc += metrics["char_accuracy_pct"]

            sample_results.append({
                "sample_id": sample["id"],
                "ground_truth": gt_text,
                "prediction": pred_text,
                "metrics": metrics,
            })

            print(f"  [{sample['id']}] CER: {metrics['cer']:.2f} | WER: {metrics['wer']:.2f} | Acc: {metrics['char_accuracy_pct']}%")
            print(f"       GT  : {gt_text}")
            print(f"       PRED: {pred_text}")

        n = len(samples)
        avg_cer = round(total_cer / n, 4)
        avg_wer = round(total_wer / n, 4)
        avg_acc = round(total_acc / n, 2)

        pipeline_summaries[mode_key] = {
            "name": mode_name,
            "average_cer": avg_cer,
            "average_wer": avg_wer,
            "average_char_accuracy": avg_acc,
            "samples": sample_results,
        }

        print(f"-> Summary for {mode_key}: Avg CER = {avg_cer} | Avg WER = {avg_wer} | Avg Accuracy = {avg_acc}%")

    # Step 4: Model Configuration Optimization for Handwriting Slant & Links
    print("\n" + "=" * 75)
    print("Evaluating Model Optimization: Enhanced Grayscale + Handwriting Slant Tuning")
    print("=" * 75)

    tuned_cer = 0.0
    tuned_wer = 0.0
    tuned_acc = 0.0
    tuned_samples = []

    for sample in samples:
        img_path = os.path.join("dataset", "input_images", sample["image_file"])
        gt_text = sample["ground_truth_text"]

        _, processed_img = preprocess_image(img_path, pipeline_mode="enhanced_grayscale")

        # Tuned parameters: lower text threshold, allow higher slant tolerance, lower link threshold
        raw_detections = reader.readtext(
            processed_img,
            detail=1,
            paragraph=False,
            text_threshold=0.55,   # Default 0.7: lower threshold to capture lighter handwritten strokes
            low_text=0.30,         # Default 0.4: captures fine letter loops
            link_threshold=0.35,   # Default 0.4: helps link cursive letters
            slope_ths=0.25,        # Default 0.1: allows slanted handwriting lines
            width_ths=0.6,         # Default 0.5: accommodates wider handwritten word gaps
        )
        sorted_detections = sort_detections_in_reading_order(raw_detections)
        extracted = clean_and_assemble_text(sorted_detections)
        pred_text = extracted["text"]

        metrics = compute_metrics(gt_text, pred_text)
        tuned_cer += metrics["cer"]
        tuned_wer += metrics["wer"]
        tuned_acc += metrics["char_accuracy_pct"]

        tuned_samples.append({
            "sample_id": sample["id"],
            "ground_truth": gt_text,
            "prediction": pred_text,
            "metrics": metrics,
        })
        print(f"  [{sample['id']}] CER: {metrics['cer']:.2f} | WER: {metrics['wer']:.2f} | Acc: {metrics['char_accuracy_pct']}%")
        print(f"       GT  : {gt_text}")
        print(f"       PRED: {pred_text}")

    n = len(samples)
    pipeline_summaries["tuned_handwriting_config"] = {
        "name": "Enhanced Grayscale + Cursive Slant Tuning",
        "average_cer": round(tuned_cer / n, 4),
        "average_wer": round(tuned_wer / n, 4),
        "average_char_accuracy": round(tuned_acc / n, 2),
        "samples": tuned_samples,
    }

    # Save benchmark log
    os.makedirs(os.path.join("outputs", "logs"), exist_ok=True)
    report_path = os.path.join("outputs", "logs", "accuracy_benchmark.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(pipeline_summaries, f, indent=2)

    print("\n" + "=" * 75)
    print("FINAL BENCHMARK COMPARISON TABLE")
    print("=" * 75)
    print(f"{'Pipeline Technique':<48} | {'Avg CER':<8} | {'Avg WER':<8} | {'Avg Accuracy':<10}")
    print("-" * 75)
    for key, data in pipeline_summaries.items():
        print(f"{data['name']:<48} | {data['average_cer']:<8} | {data['average_wer']:<8} | {data['average_char_accuracy']}%")

    print(f"\nDetailed evaluation metrics saved to: {report_path}\n")


if __name__ == "__main__":
    run_benchmark()
