"""
passport.py  — /infer/passport router
"""

import logging
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from models.cropper         import cropper
from models.passport_reader import passport_reader
from utils.image_utils      import bytes_to_cv2

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/infer/passport",
    summary="Passport Number OCR",
    description=(
        "Upload a passport image. Returns the detected passport number "
        "(no_paspor) ready to autofill the patient registration form."
    ),
)
async def infer_passport(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/bmp", "image/webp"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}.",
        )

    try:
        image_bytes = await file.read()
        img_bgr     = bytes_to_cv2(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    try:
        # Step 1: Crop document to 800×500 (removes background noise)
        cropped = cropper.crop(img_bgr)

        # Step 2: Detect passport number
        ocr_result = passport_reader.read(cropped)

    except Exception as e:
        logger.exception("Error during passport inference")
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    return JSONResponse({
        "status":          "ok",
        "no_paspor":       ocr_result["no_paspor"],         # → use this to autofill
        "all_candidates":  ocr_result["all_candidates"],    # → for debugging
        "method":          ocr_result["method"],
    })
