"""
ktp.py  — /infer/ktp router
"""

import logging
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from models.cropper       import cropper
from models.ktp_reader    import ktp_reader
from utils.image_utils    import bytes_to_cv2
from utils.ktp_layout     import PASIEN_FIELD_MAP

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/infer/ktp",
    summary="KTP OCR",
    description=(
        "Upload a KTP image. Returns OCR-extracted field values ready to "
        "autofill the patient registration form."
    ),
)
async def infer_ktp(file: UploadFile = File(...)):
    # ── Validate file type ────────────────────────────────────────────────────
    if file.content_type not in ("image/jpeg", "image/png", "image/bmp", "image/webp"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG, PNG, BMP, or WEBP.",
        )

    try:
        image_bytes = await file.read()
        img_bgr     = bytes_to_cv2(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    try:
        # Step 1: Detect and crop document to 800×500
        cropped = cropper.crop(img_bgr)

        # Step 2: Read all KTP fields with CRNN
        raw_fields = ktp_reader.read(cropped)

    except Exception as e:
        logger.exception("Error during KTP inference")
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    # ── Build response ────────────────────────────────────────────────────────
    # `fields` contains ALL detected fields (for display/debug)
    # `pasien_data` contains only the fields that map to the pasien DB table
    pasien_data = {
        db_col: raw_fields.get(ktp_field, "")
        for ktp_field, db_col in PASIEN_FIELD_MAP.items()
    }

    return JSONResponse({
        "status":      "ok",
        "pasien_data": pasien_data,    # → use this to autofill the form
        "fields":      raw_fields,     # → full dump for debugging
    })
