"""
ktp_layout.py
=============
Defines fixed field regions on a normalized 800x500 KTP image.

Indonesian KTP has a standardized layout. After the document is
perspective-corrected to exactly 800x500px, these bounding boxes
extract each field for OCR.

Coordinates: (x1, y1, x2, y2) on an 800x500 image.
These values were empirically tuned from the SIPVAC training dataset.
"""

# Field name → (x1, y1, x2, y2) crop region
KTP_FIELD_REGIONS = {
    "nik":            (185, 75,  780, 108),
    "nama":           (185, 110, 600, 142),
    "tempat_lahir":   (185, 145, 430, 175),
    "tanggal_lahir":  (430, 145, 620, 175),
    "jenis_kelamin":  (185, 178, 430, 208),
    "gol_darah":      (430, 178, 560, 208),
    "alamat":         (185, 212, 780, 262),
    "rt_rw":          (185, 264, 420, 294),
    "kel_desa":       (185, 297, 620, 327),
    "kecamatan":      (185, 330, 620, 360),
    "agama":          (185, 363, 480, 393),
    "status_kawin":   (185, 396, 480, 426),
    "pekerjaan":      (185, 429, 620, 459),
    "kewarganegaraan":(185, 462, 420, 492),
}

# Fields that map directly to the Supabase `pasien` table columns
# (used to build the API response)
PASIEN_FIELD_MAP = {
    "nik":           "nik",
    "nama":          "nama",
    "tanggal_lahir": "tanggal_lahir",
    "jenis_kelamin": "jenis_kelamin",
    "alamat":        "alamat",
    "pekerjaan":     "pekerjaan",
}

# Fields to return even if not stored in pasien table
# (useful for UI autofill of extra fields)
ALL_OUTPUT_FIELDS = list(KTP_FIELD_REGIONS.keys())
