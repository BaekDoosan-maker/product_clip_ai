from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
import os
import io
import json

import psycopg2
from PIL import Image
import numpy as np

load_dotenv()

ENABLE_CLIP = os.getenv("ENABLE_CLIP", "0") == "1"
MODEL_AVAILABLE = False
model = None
processor = None
device = "cpu"
if ENABLE_CLIP:
    try:
        import torch
        from transformers import CLIPProcessor, CLIPModel
        MODEL_AVAILABLE = True
    except Exception:
        MODEL_AVAILABLE = False

SRC_DB_HOST = os.getenv("SRC_DB_HOST", "localhost")
SRC_DB_PORT = int(os.getenv("SRC_DB_PORT", "5432"))
SRC_DB_NAME = os.getenv("SRC_DB_NAME", "mall")
SRC_DB_USER = os.getenv("SRC_DB_USER")
SRC_DB_PASS = os.getenv("SRC_DB_PASS")

TGT_DB_HOST = os.getenv("TGT_DB_HOST", "localhost")
TGT_DB_PORT = int(os.getenv("TGT_DB_PORT", "5433"))
TGT_DB_NAME = os.getenv("TGT_DB_NAME", "productdb")
TGT_DB_USER = os.getenv("TGT_DB_USER")
TGT_DB_PASS = os.getenv("TGT_DB_PASS")

EMBED_DIM = int(os.getenv("EMBED_DIM", "512"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))

if not SRC_DB_USER or not SRC_DB_PASS:
    raise RuntimeError("소스 DB 자격증명(SRC_DB_USER/SRC_DB_PASS)이 설정되어 있지 않습니다. 환경변수를 확인하세요.")
if not TGT_DB_USER or not TGT_DB_PASS:
    raise RuntimeError("대상 DB 자격증명(TGT_DB_USER/TGT_DB_PASS)이 설정되어 있지 않습니다. 환경변수를 확인하세요.")

app = FastAPI(title="Vector Search API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def compute_embedding_placeholder(image: Image.Image, dim: int = EMBED_DIM):
    b = io.BytesIO()
    image.save(b, format="PNG")
    seed = abs(hash(b.getvalue())) % (2 ** 32)
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    vec /= (np.linalg.norm(vec) + 1e-12)
    return vec


def compute_embedding_clip(image: Image.Image):
    global model, processor, device, MODEL_AVAILABLE

    if not ENABLE_CLIP:
        return compute_embedding_placeholder(image)

    if model is None or processor is None:
        try:
            import torch
            from transformers import CLIPProcessor, CLIPModel
            device = "cuda" if torch.cuda.is_available() else "cpu"
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            model.to(device)
            model.eval()
            MODEL_AVAILABLE = True
        except Exception:
            model = None
            processor = None
            MODEL_AVAILABLE = False
            return compute_embedding_placeholder(image)

    inputs = processor(images=image, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    vec = features[0].cpu().numpy().astype(np.float32)
    vec /= (np.linalg.norm(vec) + 1e-12)
    return vec


def vec_to_sql_literal(vec: np.ndarray) -> str:
    return "[" + ",".join(f"{float(x):.6f}" for x in vec.tolist()) + "]"


class SearchResponseItem(BaseModel):
    product_id: str
    image_url: Optional[str] = None
    score: float
    rank: Optional[int] = None


@app.get("/health")
def health():
    return {"ready": True, "model": MODEL_AVAILABLE}


@app.post("/search_similar", response_model=List[SearchResponseItem])
async def search_similar(file: UploadFile = File(...), top_k: int = DEFAULT_TOP_K):

    fname = file.filename.lower()
    if not (fname.endswith(".jpg") or fname.endswith(".jpeg") or fname.endswith(".png")):
        raise HTTPException(status_code=400, detail="Only .jpg/.jpeg/.png files are supported")

    try:
        body = await file.read()
        image = Image.open(io.BytesIO(body)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        if ENABLE_CLIP:
            vec = compute_embedding_clip(image)
        else:
            vec = compute_embedding_placeholder(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Embedding failure")

    vec_text = vec_to_sql_literal(vec)

    try:
        tgt_conn = psycopg2.connect(host=TGT_DB_HOST, port=TGT_DB_PORT, dbname=TGT_DB_NAME, user=TGT_DB_USER, password=TGT_DB_PASS)
        candidate_limit = min(max(top_k * 5, top_k), 200)
        with tgt_conn.cursor() as cur:
            sql = "SELECT product_id, vector <-> %s::vector AS distance FROM product_vectors ORDER BY distance ASC LIMIT %s"
            cur.execute(sql, (vec_text, candidate_limit))
            rows = cur.fetchall()
    except Exception as e:
        raise HTTPException(status_code=500, detail="Target DB query failed")
    finally:
        try:
            tgt_conn.close()
        except Exception:
            pass

    results = []
    if not rows:
        return results

    try:
        src_conn = psycopg2.connect(host=SRC_DB_HOST, port=SRC_DB_PORT, dbname=SRC_DB_NAME, user=SRC_DB_USER, password=SRC_DB_PASS)
        with src_conn.cursor() as cur:
            rank = 1
            for pid, dist in rows:
                try:
                    cur.execute("SELECT id, image_url, COALESCE(name, '') FROM product WHERE id = %s AND is_active = true", (pid,))
                    rec = cur.fetchone()
                except Exception:
                    rec = None

                if not rec:
                    continue

                image_url = rec[1] if len(rec) > 1 else None
                try:
                    d = float(dist)
                    cos_sim = 1.0 - (d * d) / 2.0
                    cos_sim = max(-1.0, min(1.0, cos_sim))
                    sim01 = (cos_sim + 1.0) / 2.0
                except Exception:
                    sim01 = 0.0

                results.append({"product_id": str(pid), "image_url": image_url, "score": float(sim01), "rank": rank})
                rank += 1
                if rank > top_k:
                    break
    except Exception as e:
        for pid, dist in rows:
            try:
                d = float(dist)
                cos_sim = 1.0 - (d * d) / 2.0
                cos_sim = max(-1.0, min(1.0, cos_sim))
                sim01 = (cos_sim + 1.0) / 2.0
            except Exception:
                sim01 = 0.0
            results.append({"product_id": str(pid), "image_url": None, "score": float(sim01), "rank": None})
    finally:
        try:
            src_conn.close()
        except Exception:
            pass

    return results
