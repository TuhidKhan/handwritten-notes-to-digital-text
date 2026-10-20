"""Image Preprocessing Utilities for Handwritten Text OCR.

This module provides image transformation functions using OpenCV and NumPy,
designed to optimize contrast, remove noise, and prepare handwritten notes
for deep-learning OCR inference.
"""

import base64
import cv2
import numpy as np
from PIL import Image
import io
from typing import Tuple, Optional, Union


def load_image(source: Union[str, bytes, np.ndarray, Image.Image]) -> np.ndarray:
    """Loads an image from multiple possible source types into a BGR NumPy array.

    Args:
        source: File path (str), raw image bytes (bytes), PIL Image, or NumPy array.

    Returns:
        np.ndarray: BGR image matrix.

    Raises:
        ValueError: If source is invalid or image decoding fails.
    """
    if isinstance(source, np.ndarray):
        # Already a numpy array; ensure 3 channels if colored
        if len(source.shape) == 2:
            return cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
        return source

    if isinstance(source, Image.Image):
        # Convert PIL Image to RGB NumPy array, then to BGR for OpenCV
        rgb_array = np.array(source.convert("RGB"))
        return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    if isinstance(source, bytes):
        # Decode in-memory byte buffer
        nparr = np.frombuffer(source, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image from byte buffer.")
        return img

    if isinstance(source, str):
        # Read from disk path
        img = cv2.imread(source, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image from path: {source}")
        return img

    raise ValueError(f"Unsupported image source type: {type(source)}")


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Converts a BGR image to single-channel 8-bit grayscale.

    Args:
        image: BGR image matrix.

    Returns:
        np.ndarray: Grayscale image.
    """
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_for_ocr(
    image: np.ndarray, min_height: int = 400, max_dimension: int = 2200
) -> np.ndarray:
    """Resizes image to an optimal resolution range for OCR text detection.

    Handwritten strokes that are too small blur together; images that are
    excessively large consume memory and slow down detection without accuracy gains.

    Args:
        image: Input image (grayscale or BGR).
        min_height: Minimum pixel height to upscale small crops.
        max_dimension: Maximum dimension (width or height) to downscale huge images.

    Returns:
        np.ndarray: Rescaled image.
    """
    h, w = image.shape[:2]
    
    # Upscale if image is too small
    if h < min_height:
        scale = min_height / float(h)
        new_w = int(w * scale)
        return cv2.resize(image, (new_w, min_height), interpolation=cv2.INTER_CUBIC)

    # Downscale if larger than maximum dimension
    longest = max(h, w)
    if longest > max_dimension:
        scale = max_dimension / float(longest)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return image


def apply_clahe(
    gray: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Unlike global histogram equalization, CLAHE operates on small contextual regions
    (tiles) to boost local contrast of ink strokes without amplifying global background
    shadows or lighting gradients.

    Args:
        gray: Single-channel grayscale image.
        clip_limit: Threshold for contrast limiting.
        tile_grid_size: Grid dimensions for localized histogram equalization.

    Returns:
        np.ndarray: Contrast-enhanced grayscale image.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def reduce_noise(gray: np.ndarray, method: str = "bilateral") -> np.ndarray:
    """Reduces high-frequency background noise while preserving handwritten edges.

    Args:
        gray: Single-channel grayscale image.
        method: 'bilateral' (edge-preserving) or 'gaussian'.

    Returns:
        np.ndarray: Denoised grayscale image.
    """
    if method == "bilateral":
        # Bilateral filter keeps sharp ink boundaries while smoothing paper texture
        return cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
    elif method == "gaussian":
        return cv2.GaussianBlur(gray, (5, 5), 0)
    return gray


def binarize(gray: np.ndarray, method: str = "otsu") -> np.ndarray:
    """Converts a grayscale image into a pure black and white (binarized) image.

    Args:
        gray: Single-channel grayscale image.
        method: 'otsu' (automatic global threshold) or 'adaptive' (local neighborhood threshold).

    Returns:
        np.ndarray: Binary image (pixel values 0 or 255).
    """
    if method == "otsu":
        # Automatically calculates optimum threshold by minimizing intra-class variance
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return thresh
    elif method == "adaptive":
        # Computes threshold for individual pixel regions to handle non-uniform lighting
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=15,
            C=8,
        )
    return gray


def preprocess_image(
    source: Union[str, bytes, np.ndarray, Image.Image],
    pipeline_mode: str = "enhanced_grayscale",
) -> Tuple[np.ndarray, np.ndarray]:
    """Complete image preprocessing pipeline for handwritten text extraction.

    Args:
        source: Image path, bytes, PIL Image, or NumPy array.
        pipeline_mode:
            - 'enhanced_grayscale' (Recommended for EasyOCR): Grayscale + CLAHE + Bilateral Denoising.
            - 'otsu_binarized': Grayscale + Denoising + Otsu Thresholding.
            - 'adaptive_binarized': Grayscale + Denoising + Adaptive Gaussian Thresholding.
            - 'raw': No transformation beyond standard color loading.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (original_bgr_image, processed_image)
    """
    original = load_image(source)
    resized = resize_for_ocr(original)

    if pipeline_mode == "raw":
        return original, resized

    # Step 1: Grayscale conversion
    gray = to_grayscale(resized)

    if pipeline_mode == "enhanced_grayscale":
        # Step 2: Edge-preserving noise filtration
        denoised = reduce_noise(gray, method="bilateral")
        # Step 3: Localized contrast enhancement (CLAHE)
        processed = apply_clahe(denoised, clip_limit=2.0)
        return original, processed

    elif pipeline_mode == "otsu_binarized":
        denoised = reduce_noise(gray, method="gaussian")
        processed = binarize(denoised, method="otsu")
        return original, processed

    elif pipeline_mode == "adaptive_binarized":
        denoised = reduce_noise(gray, method="gaussian")
        processed = binarize(denoised, method="adaptive")
        return original, processed

    # Default fallback
    return original, gray


def image_to_base64(image: np.ndarray, format_ext: str = ".png") -> str:
    """Encodes an OpenCV image array to a Base64 Data URL string for web display.

    Args:
        image: NumPy image matrix (BGR or grayscale).
        format_ext: Target image format extension (e.g. '.png', '.jpg').

    Returns:
        str: Data URL string formatted as 'data:image/png;base64,...'.
    """
    success, buffer = cv2.imencode(format_ext, image)
    if not success:
        return ""
    b64_string = base64.b64encode(buffer).decode("utf-8")
    mime = "image/jpeg" if format_ext.lower() in [".jpg", ".jpeg"] else "image/png"
    return f"data:{mime};base64,{b64_string}"
