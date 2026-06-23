"""
cropper.py
==========
Mask-RCNN document cropper using the trained model_final.pth.

Detects the document card (KTP or passport) in the image,
finds the 4 corner points, and returns a perspective-corrected
800x500 crop.
"""

import os
import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# detectron2 is a heavy dependency — import lazily so the service
# can still start even if detectron2 is not installed yet.
_detectron2_available = False
try:
    import torch
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    _detectron2_available = True
except ImportError:
    logger.warning("detectron2 not installed. Cropper will use fallback (full image).")

from utils.image_utils import (
    find_four_corners, order_corners, perspective_transform,
    validate_corners
)

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "..", "weights", "model_final.pth")
OUT_W, OUT_H = 800, 500
SCORE_THRESH = 0.5


def _build_cfg():
    """Build the detectron2 config matching the training setup."""
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.WEIGHTS = os.path.abspath(WEIGHTS_PATH)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH
    cfg.MODEL.DEVICE = "cuda" if (
        _detectron2_available and __import__("torch").cuda.is_available()
    ) else "cpu"

    # Anchor settings (same as training)
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32], [64], [128], [256], [512]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 1.6]]

    cfg.INPUT.MIN_SIZE_TEST = 512
    cfg.INPUT.MAX_SIZE_TEST = 512
    return cfg


class DocumentCropper:
    """
    Wraps the trained Mask-RCNN model.
    Provides a `crop(image_bgr)` method that returns the
    perspective-corrected document image (800x500 BGR).
    """

    def __init__(self):
        self._predictor = None
        self._load()

    def _load(self):
        if not _detectron2_available:
            logger.warning("DocumentCropper: running without detectron2 (fallback mode).")
            return
        if not os.path.isfile(WEIGHTS_PATH):
            logger.warning(
                f"DocumentCropper: weights not found at {WEIGHTS_PATH}. "
                "Falling back to full-image mode. "
                "Upload model_final.pth to the weights/ folder."
            )
            return
        try:
            cfg = _build_cfg()
            self._predictor = DefaultPredictor(cfg)
            logger.info("DocumentCropper: Mask-RCNN loaded successfully.")
        except Exception as e:
            logger.error(f"DocumentCropper: failed to load model — {e}")

    def crop(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Detect and crop the document from the image.

        Returns
        -------
        np.ndarray
            800x500 BGR image of the rectified document.
            Falls back to a simple resize if detection fails.
        """
        H, W = image_bgr.shape[:2]

        if self._predictor is None:
            # No model available — return a plain resize as fallback
            logger.debug("Cropper fallback: resizing to 800x500.")
            return cv2.resize(image_bgr, (OUT_W, OUT_H))

        outputs = self._predictor(image_bgr)
        instances = outputs["instances"]

        if len(instances) == 0:
            logger.debug("No document detected; falling back to resize.")
            return cv2.resize(image_bgr, (OUT_W, OUT_H))

        # Pick the detection with the highest confidence score
        scores = instances.scores.cpu().numpy()
        best_i = np.argmax(scores)
        mask = instances.pred_masks[best_i].cpu().numpy()
        bbox = instances.pred_boxes.tensor[best_i].cpu().numpy()
        x1, y1, x2, y2 = bbox.astype(int)

        corners = find_four_corners(mask)
        if corners is None:
            corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        else:
            is_valid, _, _ = validate_corners(corners, H, W)
            if not is_valid:
                corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

        return perspective_transform(image_bgr, corners, OUT_W, OUT_H)


# Module-level singleton — loaded once when the service starts
cropper = DocumentCropper()
