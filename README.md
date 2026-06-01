# FS Waste Classifier API

API FastAPI untuk klasifikasi gambar sampah menggunakan model TFLite dari artefak tim FS.

## Kelas

- Kaca
- Kardus
- Kertas
- Logam
- Plastik
- Residu

## Struktur

```text
app/
  main.py        # endpoint FastAPI
  inference.py   # loader model dan prediksi TFLite
  schemas.py     # response schema
  settings.py    # konfigurasi env
models/
  waste_classifier.tflite
  class_names.txt
```

## Jalankan Lokal

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Dokumentasi Swagger tersedia di:

```text
http://localhost:8000/docs
```

## URL API

Jika sudah deploy ke Railway, format URL-nya seperti ini:

```text
API URL   : https://web-production-287d1.up.railway.app
Predict   : POST https://web-production-287d1.up.railway.app/predict
Health    : GET  https://web-production-287d1.up.railway.app/health
Classes   : GET  https://web-production-287d1.up.railway.app/classes
```

## Endpoint

### `GET /health`

Mengecek status API dan apakah model berhasil dimuat.

Contoh response:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### `GET /model`

Mengembalikan metadata model.

Contoh response:

```json
{
  "model_type": "tflite",
  "image_size": 224,
  "classes": ["Kaca", "Kardus", "Kertas", "Logam", "Plastik", "Residu"]
}
```

### `GET /classes`

Mengembalikan daftar kelas.

Contoh response:

```json
{
  "classes": ["Kaca", "Kardus", "Kertas", "Logam", "Plastik", "Residu"]
}
```

### `POST /predict`

Upload gambar dengan field form-data bernama `file`.

Contoh cURL:

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@contoh_gambar.jpg"
```

Contoh Python client:

```bash
python sample_client.py contoh_gambar.jpg
```

Contoh response:

```json
{
  "label": "Plastik",
  "confidence": 0.91,
  "predictions": [
    { "label": "Plastik", "confidence": 0.91 },
    { "label": "Residu", "confidence": 0.04 }
  ]
}
```

## Jalankan dengan Docker

```bash
docker build -t fs-waste-api .
docker run --rm -p 8000:8000 fs-waste-api
```

## Konfigurasi Env

Semua env memakai prefix `FS_`.

```text
FS_MODEL_PATH=models/waste_classifier.tflite
FS_CLASS_NAMES_PATH=models/class_names.txt
FS_IMAGE_SIZE=224
FS_MAX_UPLOAD_MB=8
```

## Catatan Deployment

- Untuk Linux container, dependency memakai `tflite-runtime` agar image lebih ringan.
- Untuk Windows lokal, `requirements.txt` memakai `tensorflow` karena paket `tflite-runtime` umumnya tidak tersedia resmi untuk Windows.
- Endpoint `/predict` menerima JPEG, PNG, WEBP, dan format gambar lain yang didukung Pillow.
