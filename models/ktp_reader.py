"""
ktp_reader.py
=============
EasyOCR-based KTP field extractor.

Strategy:
  1. Preprocess image (upscale 2x, denoise, sharpen)
  2. Run EasyOCR to detect all text blocks with bounding boxes
  3. Group blocks into rows by y-coordinate proximity
  4. Find KTP field labels by keyword matching (with OCR-typo variants)
  5. Extract the value to the right of the label on the same row
     OR from the next row if the label is standalone
  6. Apply field-specific cleanup & normalization
  7. Return dict of field_name -> value

This approach works WITHOUT CRNN weights and WITHOUT perspective correction.
"""

import re
import logging
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_easyocr_available = False
try:
    import easyocr as _easyocr
    _easyocr_available = True
except ImportError:
    logger.warning("easyocr not installed. KTPReader will return empty strings.")

# ── Label keywords per field (OCR typo variants included) ─────────────────────
FIELD_KEYWORDS: Dict[str, List[str]] = {
    "nik":             ["NIK", "NIX", "NI K"],
    "nama":            ["NAMA"],
    "tempat_lahir":    ["TEMPAT", "TMP LAHIR", "TEMPATLAHIR"],
    "jenis_kelamin":   ["JENIS KELAMIN", "JENISKELAMIN", "JENIS", "KELAMIN"],
    "gol_darah":       ["GOL. DARAH", "GOL DARAH", "GOL.DARAH", "GOLDARAH"],
    "alamat":          ["ALAMAT"],
    "rt_rw":           ["RT/RW", "RT RW"],
    "kel_desa":        ["KEL/DESA", "KELURAHAN/DESA", "KEL.", "DESA"],
    "kecamatan":       ["KECAMATAN"],
    "agama":           ["AGAMA"],
    "status_kawin":    ["STATUS PERKAWINAN", "STATUS", "PERKAWINAN"],
    "pekerjaan":       ["PEKERJAAN"],
    "kewarganegaraan": ["KEWARGANEGARAAN"],
}

# Ordered by priority so tanggal_lahir is parsed from tempat_lahir row
FIELD_ORDER = [
    "nik", "nama", "tempat_lahir", "jenis_kelamin", "gol_darah",
    "alamat", "rt_rw", "kel_desa", "kecamatan",
    "agama", "status_kawin", "pekerjaan", "kewarganegaraan",
]

DATE_RE   = re.compile(r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b')
NIK_RE    = re.compile(r'\b(\d{16})\b')
NIK_LOOSE = re.compile(r'(\d{14,18})')   # fallback if spaced


# ── Image preprocessing ───────────────────────────────────────────────────────

def _preprocess(img_bgr: np.ndarray) -> np.ndarray:
    gray      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    upscaled  = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    denoised  = cv2.fastNlMeansDenoising(upscaled, h=10, templateWindowSize=7, searchWindowSize=21)
    kernel    = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    return sharpened


# ── Block grouping helpers ────────────────────────────────────────────────────

Block = Tuple[float, float, float, float, str, float]   # x1,y1,x2,y2,text,conf


def _center_y(b: Block) -> float:
    return (b[1] + b[3]) / 2


def _group_rows(blocks: List[Block], tol: int = 18) -> List[List[Block]]:
    if not blocks:
        return []
    sorted_b = sorted(blocks, key=lambda b: (_center_y(b), b[0]))
    rows: List[List[Block]] = []
    cur_row = [sorted_b[0]]
    cur_cy  = _center_y(sorted_b[0])
    for b in sorted_b[1:]:
        cy = _center_y(b)
        if abs(cy - cur_cy) <= tol:
            cur_row.append(b)
        else:
            rows.append(sorted(cur_row, key=lambda b: b[0]))
            cur_row = [b]
            cur_cy  = cy
    rows.append(sorted(cur_row, key=lambda b: b[0]))
    return rows


def _is_label(text: str, keywords: List[str]) -> bool:
    tu = text.upper().strip()
    return any(kw in tu for kw in keywords)


def _value_after(row: List[Block], label_x2: float) -> str:
    parts = [b[4] for b in row if b[0] > label_x2 - 10]
    return " ".join(parts).strip()


def _clean(text: str) -> str:
    return re.sub(r'^[\s:./\-]+', '', text).strip()


def _normalize_jenis_kelamin(text: str) -> str:
    t = text.upper().strip()
    if "PEREMPUAN" in t:
        return "PEREMPUAN"
    if "LAKI" in t or t in ("L", "L."):
        return "LAKI-LAKI"
    if t in ("P", "P."):
        return "PEREMPUAN"
    return text


# ── Main Reader ───────────────────────────────────────────────────────────────

class KTPReader:
    """
    EasyOCR-based KTP reader.
    No CRNN weights or perspective correction required.
    """

    def __init__(self):
        self._reader: Optional[_easyocr.Reader] = None
        self._load()

    def _load(self):
        if not _easyocr_available:
            return
        try:
            self._reader = _easyocr.Reader(['id', 'en'], gpu=False, verbose=False)
            logger.info("KTPReader: EasyOCR initialized (id+en).")
        except Exception as e:
            logger.error(f"KTPReader: EasyOCR init failed — {e}")

    def _get_blocks(self, img: np.ndarray) -> List[Block]:
        raw = self._reader.readtext(img, detail=1, paragraph=False)
        blocks: List[Block] = []
        for (bbox, text, conf) in raw:
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            x1, y1 = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)
            t = text.strip()
            if t:
                blocks.append((x1, y1, x2, y2, t, conf))
        return blocks

    def _try_find_nik(self, blocks: List[Block]) -> str:
        """Search directly for a 16-digit NIK anywhere in OCR output."""
        # Concat all text and search
        all_text = " ".join(b[4] for b in blocks)
        all_digits = re.sub(r'\s+', '', all_text)
        m = NIK_RE.search(all_digits)
        if m:
            return m.group(1)
        # Try removing spaces within each block
        for b in blocks:
            compact = re.sub(r'\s+', '', b[4])
            m = NIK_LOOSE.search(compact)
            if m:
                return m.group(1)
        return ""

    def read(self, img_bgr: np.ndarray) -> Dict[str, str]:
        """
        Extract all KTP fields from the image.

        Returns
        -------
        dict  {field_name: value_string}
        """
        empty = {f: "" for f in list(FIELD_KEYWORDS.keys()) + ["tanggal_lahir"]}

        if self._reader is None:
            return empty

        try:
            preprocessed = _preprocess(img_bgr)
            blocks = self._get_blocks(preprocessed)
        except Exception as e:
            logger.error(f"KTPReader: OCR failed — {e}")
            return empty

        rows = _group_rows(blocks)
        results: Dict[str, str] = {f: "" for f in list(FIELD_KEYWORDS.keys()) + ["tanggal_lahir"]}

        # ── 1. Direct NIK detection ───────────────────────────────────────────
        results["nik"] = self._try_find_nik(blocks)

        # ── 2. Label-based extraction ─────────────────────────────────────────
        found_fields = set()
        if results["nik"]:
            found_fields.add("nik")

        for i, row in enumerate(rows):
            row_text = " ".join(b[4] for b in row).upper()

            for field in FIELD_ORDER:
                if field in found_fields:
                    continue
                keywords = FIELD_KEYWORDS.get(field, [])
                if not keywords:
                    continue

                for j, block in enumerate(row):
                    if not _is_label(block[4], keywords):
                        continue

                    label_x2 = block[2]

                    # Value = everything to the right of this label in the same row
                    value = _clean(_value_after(row, label_x2))

                    # If nothing to the right, check the next row
                    if not value and i + 1 < len(rows):
                        next_row = rows[i + 1]
                        next_text = " ".join(b[4] for b in next_row).upper()
                        # Only take next row if it doesn't look like another label
                        is_next_a_label = any(
                            _is_label(b[4], kws)
                            for f2, kws in FIELD_KEYWORDS.items()
                            for b in next_row
                        )
                        if not is_next_a_label:
                            value = _clean(" ".join(b[4] for b in next_row))

                    if value:
                        results[field] = value
                        found_fields.add(field)
                    break  # matched this field, move to next field

        # ── 3. Parse tanggal_lahir from tempat_lahir row ──────────────────────
        # Indonesian KTP: "Tempat/Tgl Lahir : KOTA, DD-MM-YYYY"
        tl_val = results.get("tempat_lahir", "")
        if tl_val:
            m = DATE_RE.search(tl_val)
            if m:
                results["tanggal_lahir"] = m.group(1)
                # Remove the date from tempat_lahir
                results["tempat_lahir"] = _clean(tl_val[:m.start()].rstrip(", "))
        else:
            # Fallback: search date pattern across all OCR text if not found yet
            all_text = " ".join(b[4] for b in blocks)
            m = DATE_RE.search(all_text)
            if m:
                results["tanggal_lahir"] = m.group(1)

        # ── 4. Normalize jenis_kelamin ────────────────────────────────────────
        if results.get("jenis_kelamin"):
            results["jenis_kelamin"] = _normalize_jenis_kelamin(results["jenis_kelamin"])

        # ── 5. Fallback: collect leftover text for missing multi-line fields ──
        # alamat can span multiple rows — collect rows between alamat and rt_rw labels
        if not results.get("alamat"):
            alamat_parts = []
            collecting   = False
            for row in rows:
                row_text_up = " ".join(b[4] for b in row).upper()
                if _is_label(" ".join(b[4] for b in row), FIELD_KEYWORDS["alamat"]):
                    collecting = True
                    # check if value is inline
                    for b in row:
                        if _is_label(b[4], FIELD_KEYWORDS["alamat"]):
                            inline = _clean(_value_after(row, b[2]))
                            if inline:
                                alamat_parts.append(inline)
                    continue
                if collecting:
                    if any(_is_label(b[4], FIELD_KEYWORDS["rt_rw"]) for b in row):
                        break
                    # Stop if we hit another known label
                    if any(
                        _is_label(b[4], kws)
                        for f, kws in FIELD_KEYWORDS.items()
                        if f not in ("alamat",)
                        for b in row
                    ):
                        break
                    alamat_parts.append(" ".join(b[4] for b in row))
            if alamat_parts:
                results["alamat"] = " ".join(alamat_parts).strip()

        logger.info(f"KTPReader result: {results}")
        return results


# Singleton — loaded once at service startup
ktp_reader = KTPReader()
