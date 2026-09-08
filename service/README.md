# Face Verification API

Backend-only. No UI, no auth, no database. Images are processed in memory and never written to disk.

Pretrained Inception-ResNet-v1 (VGGFace2, 512-d embeddings) + MTCNN detection/alignment,
via `facenet-pytorch` (a maintained PyTorch port of davidsandberg/facenet weights).

## Run

```bash
pip install -r requirements.txt
FACE_MATCH_THRESHOLD=1.0 uvicorn main:app --host 0.0.0.0 --port 8000
```

## Endpoint

`POST /verify` — multipart form with `reference_image` and `selfie_image`.

```bash
curl -F reference_image=@ref.jpg -F selfie_image=@selfie.jpg http://localhost:8000/verify
```

Response: `{"verified": true, "distance": 0.6123}`

Errors with HTTP 400 when an image has zero or multiple faces, or is unreadable.

## Config

| Variable | Default | Meaning |
| --- | --- | --- |
| `FACE_MATCH_THRESHOLD` | `1.0` | Max Euclidean distance counted as a match |
