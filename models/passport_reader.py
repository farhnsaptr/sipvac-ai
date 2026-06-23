"""
passport_reader.py
==================
Detects the passport number from a passport image.

Strategy (same as ai-result/training/passport_detector.py):
  1. Preprocess (grayscale, denoise, sharpen, adaptive threshold)
  2. Crop the MRZ zone (bottom 28% of the image)
  3. EasyOCR on full image + MRZ zone
  4. Pytesseract as fallback
  5. Regex patterns to find passport number candidates
  6. TD3 MRZ Line 2 parsing for highest-confidence result
"""

import re
import os
import logging
import numpy as np
import cv2
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

# ── Optional dependencies ─────────────────────────────────────────────────────
_easyocr_available = False
try:
    import easyocr as _easyocr
    _easyocr_available = True
except ImportError:
    logger.warning("easyocr not installed. Passport OCR will use tesseract only.")

_tesseract_available = False
try:
    import pytesseract
    # On HuggingFace Spaces, tesseract is installed system-wide
    _tesseract_available = True
except ImportError:
    logger.warning("pytesseract not installed.")

# ── Passport number regex patterns (most-specific first) ─────────────────────
PASSPORT_PATTERNS = [
    r"\b[A-Z][0-9]{7}\b",          # Indonesia / common:  A1234567
    r"\b[A-Z]{2}[0-9]{7}\b",       # Two letters + 7 digits
    r"\b[A-Z]{1,2}[0-9]{6,9}\b",   # General 1-2 letters + 6-9 digits
    r"\b[0-9]{8,9}\b",              # Numeric-only (some countries)
]


# ── Image preprocessing ───────────────────────────────────────────────────────

def _preprocess(img_bgr: np.ndarray) -> dict:
    gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    kernel   = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    adaptive  = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8
    )
    upscaled  = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return {"gray": gray, "sharpened": sharpened, "adaptive": adaptive, "upscaled": upscaled}


def _extract_mrz_zone(img_bgr: np.ndarray, zone_ratio: float = 0.28) -> np.ndarray:
    h     = img_bgr.shape[0]
    start = int(h * (1.0 - zone_ratio))
    return img_bgr[start:, :]


# ── MRZ parsing ───────────────────────────────────────────────────────────────

def _parse_mrz_line2(line: str) -> Optional[str]:
    """Parse TD3 MRZ Line 2 — positions 0-8 are the document number."""
    clean = re.sub(r"[^A-Z0-9<]", "", line.upper())
    if len(clean) < 44:
        return None
    doc_num = clean[:9].rstrip("<")
    if len(doc_num) >= 6 and re.match(r"^[A-Z0-9]+$", doc_num):
        return doc_num
    return None


def _extract_from_mrz_text(mrz_text: str) -> Optional[str]:
    for line in mrz_text.upper().split("\n"):
        clean = re.sub(r"\s+", "", line)
        if len(clean) >= 30 and ("<" in clean or re.search(r"[A-Z]{2,}", clean)):
            result = _parse_mrz_line2(clean)
            if result:
                return result
    return None


def _find_candidates(text: str) -> List[str]:
    text_upper = text.upper()
    found: List[str] = []
    for pattern in PASSPORT_PATTERNS:
        for m in re.findall(pattern, text_upper):
            if m not in found:
                found.append(m)
    return found


# ── Main reader class ─────────────────────────────────────────────────────────

class PassportReader:
    """
    Reads a passport image and extracts the passport number.
    EasyOCR reader is initialized once and reused across requests.
    """

    def __init__(self):
        self._easy_reader = None
        self._load()

    def _load(self):
        if not _easyocr_available:
            return
        try:
            # gpu=False is safe on CPU-only HF Spaces
            self._easy_reader = _easyocr.Reader(["en"], gpu=False, verbose=False)
            logger.info("PassportReader: EasyOCR initialized.")
        except Exception as e:
            logger.error(f"PassportReader: EasyOCR init failed — {e}")

    def read(self, img_bgr: np.ndarray) -> Dict:
        """
        Parameters
        ----------
        img_bgr : np.ndarray
            Full-size BGR passport image (already loaded from bytes).

        Returns
        -------
        dict with keys:
            no_paspor       – best detected passport number (str or None)
            all_candidates  – all candidate strings found
            method          – "MRZ Parse" | "Regex Pattern" | None
        """
        result = {
            "no_paspor":      None,
            "all_candidates": [],
            "method":         None,
        }

        preprocessed = _preprocess(img_bgr)
        mrz_bgr  = _extract_mrz_zone(img_bgr)
        mrz_gray = cv2.cvtColor(mrz_bgr, cv2.COLOR_BGR2GRAY)

        candidates: List[str] = []
        mrz_num_best: Optional[str] = None

        # ── EasyOCR ───────────────────────────────────────────────────────────
        if self._easy_reader is not None:
            try:
                easy_full = self._easy_reader.readtext(img_bgr, detail=0, paragraph=False)
                full_text = " ".join(easy_full)
                candidates = _find_candidates(full_text)

                easy_mrz = self._easy_reader.readtext(
                    mrz_gray, detail=0, paragraph=False,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<",
                )
                mrz_text  = "\n".join(easy_mrz)
                mrz_num_best = _extract_from_mrz_text(mrz_text)
                if mrz_num_best and mrz_num_best not in candidates:
                    candidates.insert(0, mrz_num_best)
            except Exception as e:
                logger.warning(f"PassportReader: EasyOCR error — {e}")

        # ── Pytesseract fallback ──────────────────────────────────────────────
        if _tesseract_available:
            try:
                cfg_full = "--oem 3 --psm 6"
                cfg_mrz  = r"--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"

                tess_full = pytesseract.image_to_string(preprocessed["sharpened"], config=cfg_full)
                tess_mrz  = pytesseract.image_to_string(mrz_gray, config=cfg_mrz)

                tess_cands = _find_candidates(tess_full)
                mrz_tess   = _extract_from_mrz_text(tess_mrz)
                if mrz_tess and mrz_tess not in tess_cands:
                    tess_cands.insert(0, mrz_tess)
                    if mrz_num_best is None:
                        mrz_num_best = mrz_tess

                for c in tess_cands:
                    if c not in candidates:
                        candidates.append(c)
            except Exception as e:
                logger.warning(f"PassportReader: Tesseract error — {e}")

        result["all_candidates"] = candidates
        if candidates:
            result["no_paspor"] = candidates[0]
            result["method"] = (
                "MRZ Parse" if mrz_num_best and candidates[0] == mrz_num_best
                else "Regex Pattern"
            )

        return result


# Singleton
passport_reader = PassportReader()
