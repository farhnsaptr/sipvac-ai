"""
main.py
=======
SIPVAC AI Service — FastAPI entry point.

Designed for deployment on HuggingFace Spaces (Docker SDK).

Endpoints:
  POST /infer/ktp       — KTP OCR (returns all field values)
  POST /infer/passport  — Passport OCR (returns passport number)
  GET  /health          — Health check

CORS is wide-open so the React frontend can call this directly.
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.ktp      import router as ktp_router
from routers.passport import router as passport_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

app = FastAPI(
    title="SIPVAC AI Service",
    description=(
        "OCR inference service for SIPVAC. "
        "Extracts KTP field data and passport numbers from uploaded images."
    ),
    version="1.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json",
)

# ── CORS: allow the React frontend to call this service directly ──────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(ktp_router,      tags=["KTP OCR"])
app.include_router(passport_router, tags=["Passport OCR"])


@app.get("/health", tags=["Health"])
def health():
    """Quick health check — returns 200 if the service is up."""
    return {"status": "ok", "service": "sipvac-ai"}


@app.get("/", tags=["Health"])
def root():
    return {"message": "SIPVAC AI Service is running. See /docs for API reference."}


# ── Local dev ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=True)
