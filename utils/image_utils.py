"""
image_utils.py
==============
Shared image preprocessing helpers used by both the KTP and passport pipelines.
"""

import io
import cv2
import numpy as np
from PIL import Image


def bytes_to_cv2(image_bytes: bytes) -> np.ndarray:
    """Convert raw image bytes to a BGR OpenCV array."""
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Make sure it is a valid JPG/PNG/BMP.")
    return img


def cv2_to_pil(img_bgr: np.ndarray) -> Image.Image:
    """Convert BGR OpenCV array to PIL Image (RGB)."""
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def pil_to_cv2(img_pil: Image.Image) -> np.ndarray:
    """Convert PIL Image (RGB) to BGR OpenCV array."""
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def resize_pad_grayscale(img_bgr: np.ndarray, target_h: int = 32, target_w: int = 512) -> np.ndarray:
    """
    Convert to grayscale, resize to target_h while keeping aspect ratio,
    then right-pad to target_w with white (255).
    Returns a (target_h, target_w) uint8 array.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    new_w = min(int(w * target_h / h), target_w)
    resized = cv2.resize(gray, (new_w, target_h), interpolation=cv2.INTER_CUBIC)
    padded = np.full((target_h, target_w), 255, dtype=np.uint8)
    padded[:, :new_w] = resized
    return padded


def find_four_corners(mask: np.ndarray):
    """
    Given a binary mask (bool or uint8), find the 4 corners of the document.
    Returns a (4, 2) float32 array or None if not found.
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
    if len(approx) == 4:
        return approx.reshape(4, 2).astype(np.float32)
    rect = cv2.minAreaRect(largest)
    return cv2.boxPoints(rect).astype(np.float32)


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 corner points as [top-left, top-right, bottom-right, bottom-left]."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]     # top-left
    rect[2] = pts[np.argmax(s)]     # bottom-right
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def perspective_transform(image: np.ndarray, corners: np.ndarray,
                          out_w: int = 800, out_h: int = 500) -> np.ndarray:
    """Apply perspective transform to rectify the document."""
    src = order_corners(corners)
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32
    )
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (out_w, out_h))


def validate_corners(corners: np.ndarray, img_h: int, img_w: int,
                     min_area_ratio: float = 0.05) -> tuple:
    """Check if detected corners form a valid document region."""
    x, y = corners[:, 0], corners[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    ratio = area / (img_h * img_w)
    spread = max(
        np.linalg.norm(corners[i] - corners[j])
        for i in range(4) for j in range(i + 1, 4)
    )
    valid = (ratio >= min_area_ratio) and (spread >= min(img_h, img_w) * 0.3)
    return valid, ratio, spread
