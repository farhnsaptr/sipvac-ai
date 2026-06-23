"""
ktp_reader.py
=============
CRNN + Joint CTC-Attention model for reading individual KTP fields.

Architecture is identical to the training code in:
  ai-result/training/cp_training_crnn.py

Workflow:
  1. Receive the 800x500 cropped KTP image.
  2. Extract each field region using fixed bounding boxes (ktp_layout.py).
  3. Run each crop through the CRNN to decode the text.
  4. Return a dict of field_name → decoded_text.
"""

import os
import json
import logging
import numpy as np
import cv2
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_torch_available = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from PIL import Image, ImageOps
    import torchvision.transforms as T
    _torch_available = True
except ImportError:
    logger.warning("PyTorch not installed. KTPReader will return empty strings.")

from utils.image_utils import resize_pad_grayscale
from utils.ktp_layout import KTP_FIELD_REGIONS

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "..", "weights", "best_model.pth")
VOCAB_PATH   = os.path.join(os.path.dirname(__file__), "..", "weights", "vocab.json")

IMG_H, IMG_W = 32, 512


# ─── Model Architecture (mirrors training code exactly) ───────────────────────

class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True),
        )

    def forward(self, x):
        out = self.cnn(x).squeeze(2)   # [B, 512, W']
        return out.permute(2, 0, 1)    # [W', B, 512]


class BLSTMEncoder(nn.Module):
    def __init__(self, input_size=512, hidden_size=256, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=False, bidirectional=True,
                            dropout=0.1 if num_layers > 1 else 0.0)
        self.proj = nn.Linear(hidden_size * 2, hidden_size * 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.proj(out)


class CTCHead(nn.Module):
    def __init__(self, input_size=512, num_classes=100):
        super().__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, enc_out):
        return F.log_softmax(self.fc(enc_out), dim=2)


class AttentionDecoder(nn.Module):
    def __init__(self, enc_hidden=512, dec_hidden=256, num_classes=100,
                 max_len=64, sos_idx=1, eos_idx=2):
        super().__init__()
        self.num_classes = num_classes
        self.max_len = max_len
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.embed = nn.Embedding(num_classes, dec_hidden)
        self.lstm  = nn.LSTMCell(dec_hidden + enc_hidden, dec_hidden)
        self.attn_W_enc = nn.Linear(enc_hidden, dec_hidden, bias=False)
        self.attn_W_dec = nn.Linear(dec_hidden, dec_hidden, bias=False)
        self.attn_v     = nn.Linear(dec_hidden, 1, bias=False)
        self.fc = nn.Linear(dec_hidden + enc_hidden, num_classes)

    def attention(self, enc_out, dec_h):
        energy = torch.tanh(
            self.attn_W_enc(enc_out) + self.attn_W_dec(dec_h).unsqueeze(0)
        )
        score   = self.attn_v(energy).squeeze(2)
        weight  = F.softmax(score, dim=0)
        context = (weight.unsqueeze(2) * enc_out).sum(0)
        return context, weight

    def forward(self, enc_out, label_ids=None, teacher_forcing=False):
        T, B, enc_h = enc_out.shape
        max_len = label_ids.shape[1] - 1 if label_ids is not None else self.max_len
        h = torch.zeros(B, self.lstm.hidden_size, device=enc_out.device)
        c = torch.zeros(B, self.lstm.hidden_size, device=enc_out.device)
        inp = torch.full((B,), self.sos_idx, dtype=torch.long, device=enc_out.device)
        outputs = []
        for t in range(max_len):
            emb     = self.embed(inp)
            ctx, _  = self.attention(enc_out, h)
            h, c    = self.lstm(torch.cat([emb, ctx], dim=1), (h, c))
            logit   = self.fc(torch.cat([h, ctx], dim=1))
            outputs.append(logit)
            inp = logit.argmax(dim=1)
        return torch.stack(outputs, dim=1)


class CRNNJointCTCAttention(nn.Module):
    def __init__(self, num_classes, rnn_hidden=256, rnn_layers=2,
                 max_label_len=64, sos_idx=1, eos_idx=2, lambda_ctc=0.5):
        super().__init__()
        self.cnn     = VGGFeatureExtractor()
        self.encoder = BLSTMEncoder(512, rnn_hidden, rnn_layers)
        enc_out_size = rnn_hidden * 2
        self.ctc_head = CTCHead(enc_out_size, num_classes)
        self.attn_dec = AttentionDecoder(enc_out_size, rnn_hidden, num_classes,
                                         max_label_len, sos_idx, eos_idx)

    def forward(self, images, label_ids=None, teacher_forcing=False):
        feat    = self.cnn(images)
        enc_out = self.encoder(feat)
        ctc_log = self.ctc_head(enc_out)
        attn_out = self.attn_dec(enc_out, label_ids, teacher_forcing)
        return ctc_log, attn_out


# ─── Reader Class ─────────────────────────────────────────────────────────────

class KTPReader:
    """
    Loads the CRNN model once at startup and provides a `read(cropped_ktp_bgr)`
    method that returns a dict of all decoded KTP fields.
    """

    def __init__(self):
        self._model  = None
        self._vocab  = None
        self._device = None
        self._load()

    def _load(self):
        if not _torch_available:
            return

        if not os.path.isfile(WEIGHTS_PATH):
            logger.warning(
                f"KTPReader: weights not found at {WEIGHTS_PATH}. "
                "Fields will be returned as empty strings. "
                "Upload best_model.pth and vocab.json to weights/."
            )
            return

        if not os.path.isfile(VOCAB_PATH):
            logger.warning(f"KTPReader: vocab.json not found at {VOCAB_PATH}.")
            return

        try:
            with open(VOCAB_PATH, "r", encoding="utf-8") as f:
                self._vocab = json.load(f)

            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            ckpt = torch.load(WEIGHTS_PATH, map_location=self._device)
            cfg  = ckpt.get("config", {})

            self._model = CRNNJointCTCAttention(
                num_classes   = self._vocab["num_classes"],
                rnn_hidden    = cfg.get("rnn_hidden", 256),
                rnn_layers    = cfg.get("rnn_layers", 2),
                max_label_len = cfg.get("max_label_len", 64),
                sos_idx       = self._vocab["SOS_IDX"],
                eos_idx       = self._vocab["EOS_IDX"],
                lambda_ctc    = cfg.get("lambda_ctc", 0.5),
            ).to(self._device)

            self._model.load_state_dict(ckpt["model_state"])
            self._model.eval()
            logger.info(f"KTPReader: CRNN loaded on {self._device}.")

        except Exception as e:
            logger.error(f"KTPReader: failed to load model — {e}")
            self._model = None

    @torch.no_grad()
    def _decode_crop(self, crop_bgr: np.ndarray) -> str:
        """Run CRNN on a single field crop and return the decoded string."""
        if self._model is None or self._vocab is None:
            return ""

        # Preprocess: grayscale → resize/pad → tensor [-1, 1]
        padded = resize_pad_grayscale(crop_bgr, IMG_H, IMG_W)
        pil    = Image.fromarray(padded)
        t      = T.ToTensor()(pil)
        t      = (t - 0.5) / 0.5
        t      = t.unsqueeze(0).to(self._device)   # [1, 1, H, W]

        _, attn_out = self._model(t, None, teacher_forcing=False)

        idx2char = self._vocab["idx2char"]
        eos_idx  = self._vocab["EOS_IDX"]
        special  = {"<BLANK>", "<SOS>", "<EOS>", "<UNK>"}

        pred_ids = attn_out.argmax(dim=2)[0].tolist()
        result   = ""
        for idx in pred_ids:
            if idx == eos_idx:
                break
            c = idx2char.get(str(idx), "")
            if c not in special:
                result += c
        return result.strip()

    def read(self, cropped_ktp_bgr: np.ndarray) -> Dict[str, str]:
        """
        Extract all field regions from the 800x500 KTP image and decode each.

        Returns
        -------
        dict
            {field_name: decoded_text, ...}
        """
        results: Dict[str, str] = {}
        H, W = cropped_ktp_bgr.shape[:2]

        for field, (x1, y1, x2, y2) in KTP_FIELD_REGIONS.items():
            # Clamp to image bounds
            x1c = max(0, x1); y1c = max(0, y1)
            x2c = min(W, x2); y2c = min(H, y2)

            if x2c <= x1c or y2c <= y1c:
                results[field] = ""
                continue

            crop = cropped_ktp_bgr[y1c:y2c, x1c:x2c]

            if crop.size == 0:
                results[field] = ""
                continue

            try:
                results[field] = self._decode_crop(crop)
            except Exception as e:
                logger.warning(f"KTPReader: error decoding field '{field}' — {e}")
                results[field] = ""

        return results


# Singleton loaded once at startup
ktp_reader = KTPReader()
