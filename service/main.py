import io
import os

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

THRESHOLD = float(os.environ.get("FACE_MATCH_THRESHOLD", "1.0"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

mtcnn = MTCNN(image_size=160, margin=0, keep_all=True, device=DEVICE)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)

app = FastAPI(title="Face Verification API")


def embed(upload: UploadFile, label: str) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(upload.file.read())).convert("RGB")
    except Exception:
        raise HTTPException(400, f"{label}: invalid image")

    boxes, probs = mtcnn.detect(img)
    count = 0 if boxes is None else len(boxes)
    if count != 1:
        raise HTTPException(400, f"{label}: expected exactly one face, found {count}")

    face = mtcnn.extract(img, boxes, save_path=None)  # aligned 160x160, normalized
    with torch.no_grad():
        return resnet(face.to(DEVICE))[0].cpu().numpy()


@app.post("/verify")
async def verify(reference_image: UploadFile = File(...), selfie_image: UploadFile = File(...)):
    a = embed(reference_image, "reference_image")
    b = embed(selfie_image, "selfie_image")
    distance = float(np.linalg.norm(a - b))
    return {"verified": distance < THRESHOLD, "distance": round(distance, 4)}
