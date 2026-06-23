---
title: SIPVAC AI
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# SIPVAC AI Service

OCR inference microservice for the **SIPVAC** vaccination management system.

Extracts structured data from KTP (Indonesian ID card) and passport images
using a trained Mask-RCNN + CRNN pipeline.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/infer/ktp` | Upload KTP image → returns all field values |
| `POST` | `/infer/passport` | Upload passport image → returns passport number |
| `GET`  | `/health` | Health check |
| `GET`  | `/docs` | Swagger UI |

## Model Files

Upload these to the `weights/` folder in the Space:

| File | Size | Source |
|------|------|--------|
| `model_final.pth` | ~351 MB | `ai-result/preprocessing/model/` |
| `best_model.pth` | ~124 MB | `ai-result/training/model/` |
| `vocab.json` | small | Generated during CRNN training |

## Usage (Frontend)

```js
// Scan KTP
const formData = new FormData();
formData.append('file', ktpFile);
const res = await fetch('https://your-space.hf.space/infer/ktp', {
  method: 'POST',
  body: formData,
});
const { pasien_data } = await res.json();
// pasien_data → { nik, nama, tanggal_lahir, jenis_kelamin, alamat, pekerjaan }

// Scan Passport
const formData2 = new FormData();
formData2.append('file', passportFile);
const res2 = await fetch('https://your-space.hf.space/infer/passport', {
  method: 'POST',
  body: formData2,
});
const { no_paspor } = await res2.json();
```
