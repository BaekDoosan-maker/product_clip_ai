from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
import os
import io
import json
import logging
import requests

import psycopg2
from PIL import Image
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

ENABLE_CLIP = os.getenv("ENABLE_CLIP", "0") == "1"
MODEL_AVAILABLE = False
model = None
processor = None
device = "cpu"
torch = None
CLIPProcessor = None
CLIPModel = None

try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    import sys
    logger.info(f"[초기화] torch 모듈 로드 성공 - Python 경로: {sys.executable}, torch 버전: {torch.__version__}")
except Exception as e:
    import sys
    torch = None
    CLIPProcessor = None
    CLIPModel = None
    logger.warning(f"[초기화] torch 모듈 로드 실패: {e} - Python 경로: {sys.executable}")
    if ENABLE_CLIP:
        logger.warning(f"[초기화] 가상환경에서 torch를 설치하세요: pip install torch --index-url https://download.pytorch.org/whl/cpu")

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
    logger.info(f"[벡터화] Placeholder 모드 사용 (차원: {dim})")
    b = io.BytesIO()
    image.save(b, format="PNG")
    seed = abs(hash(b.getvalue())) % (2 ** 32)
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    vec /= (np.linalg.norm(vec) + 1e-12)
    
    # 벡터 통계 정보 로깅
    vec_norm = np.linalg.norm(vec)
    logger.info(f"[벡터화 완료] Placeholder - 차원: {len(vec)}, 노름: {vec_norm:.6f}, "
                f"평균: {vec.mean():.6f}, 최소: {vec.min():.6f}, 최대: {vec.max():.6f}")
    return vec


def compute_embedding_clip(image: Image.Image):
    global model, processor, device, MODEL_AVAILABLE

    if not ENABLE_CLIP:
        logger.info("[벡터화] ENABLE_CLIP=0, Placeholder로 폴백")
        return compute_embedding_placeholder(image)

    if model is None or processor is None:
        if torch is None or CLIPProcessor is None or CLIPModel is None:
            logger.error("[벡터화] torch 또는 transformers 모듈이 로드되지 않았습니다. Placeholder로 폴백")
            return compute_embedding_placeholder(image)
        
        try:
            logger.info("[벡터화] CLIP 모델 로딩 시작...")
            import sys
            logger.info(f"[벡터화] Python 경로: {sys.executable}, torch 버전: {torch.__version__}")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"[벡터화] 디바이스: {device}")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            model.to(device)
            model.eval()
            MODEL_AVAILABLE = True
            logger.info("[벡터화] CLIP 모델 로딩 완료")
        except Exception as e:
            import sys
            logger.error(f"[벡터화] CLIP 모델 로딩 실패: {e}, Placeholder로 폴백")
            logger.error(f"[벡터화] Python 경로: {sys.executable}")
            model = None
            processor = None
            MODEL_AVAILABLE = False
            return compute_embedding_placeholder(image)

    try:
        logger.info(f"[벡터화] CLIP 모델로 벡터화 시작 (이미지 크기: {image.size})")
        inputs = processor(images=image, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            features = model.get_image_features(**inputs)
        vec = features[0].cpu().numpy().astype(np.float32)
        vec /= (np.linalg.norm(vec) + 1e-12)
        
        # 벡터 통계 정보 로깅
        vec_norm = np.linalg.norm(vec)
        logger.info(f"[벡터화 완료] CLIP - 차원: {len(vec)}, 노름: {vec_norm:.6f}, "
                    f"평균: {vec.mean():.6f}, 최소: {vec.min():.6f}, 최대: {vec.max():.6f}")
        return vec
    except Exception as e:
        logger.error(f"[벡터화] CLIP 벡터화 중 오류 발생: {e}")
        raise


def vec_to_sql_literal(vec: np.ndarray) -> str:
    return "[" + ",".join(f"{float(x):.6f}" for x in vec.tolist()) + "]"


class SearchResponseItem(BaseModel):
    product_id: str
    image_url: Optional[str] = None
    score: float
    rank: Optional[int] = None


class SearchByUrlRequest(BaseModel):
    image_url: str
    top_k: Optional[int] = DEFAULT_TOP_K


@app.get("/health")
def health():
    return {"ready": True, "model": MODEL_AVAILABLE}


def load_image_from_url_or_file(image_source) -> Image.Image:
    """이미지 URL 또는 파일에서 이미지 로드"""
    if isinstance(image_source, str):
        # URL인 경우
        try:
            logger.info(f"[이미지 다운로드] URL: {image_source}")
            response = requests.get(image_source, timeout=10, stream=True)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            logger.info(f"[이미지 다운로드 성공] 크기: {image.size}")
            return image
        except Exception as e:
            logger.error(f"[이미지 다운로드 실패] {image_source}: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to download image from URL: {e}")
    else:
        # 파일인 경우
        return image_source


@app.post("/search_similar", response_model=List[SearchResponseItem])
async def search_similar(file: UploadFile = File(...), top_k: int = DEFAULT_TOP_K):

    logger.info(f"[검색 요청] 파일명: {file.filename}, top_k: {top_k}, ENABLE_CLIP: {ENABLE_CLIP}, MODEL_AVAILABLE: {MODEL_AVAILABLE}")
    
    fname = file.filename.lower()
    if not (fname.endswith(".jpg") or fname.endswith(".jpeg") or fname.endswith(".png")):
        raise HTTPException(status_code=400, detail="Only .jpg/.jpeg/.png files are supported")

    try:
        body = await file.read()
        image = Image.open(io.BytesIO(body)).convert("RGB")
        logger.info(f"[이미지 로드] 크기: {image.size}, 모드: {image.mode}, 파일 크기: {len(body)} bytes")
    except Exception as e:
        logger.error(f"[이미지 로드 실패] {e}")
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    return await _perform_search(image, top_k)


@app.post("/search_similar_by_url", response_model=List[SearchResponseItem])
async def search_similar_by_url(request: SearchByUrlRequest):
    """이미지 URL로 유사 제품 검색 (CORS 문제 해결용)"""
    logger.info(f"[검색 요청] 이미지 URL: {request.image_url}, top_k: {request.top_k}, ENABLE_CLIP: {ENABLE_CLIP}, MODEL_AVAILABLE: {MODEL_AVAILABLE}")
    
    try:
        image = load_image_from_url_or_file(request.image_url)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[이미지 로드 실패] {e}")
        raise HTTPException(status_code=400, detail=f"Failed to load image: {e}")
    
    return await _perform_search(image, request.top_k)


async def _perform_search(image: Image.Image, top_k: int) -> List[SearchResponseItem]:

    try:
        if ENABLE_CLIP:
            vec = compute_embedding_clip(image)
        else:
            logger.warning(f"[벡터화] ENABLE_CLIP이 비활성화되어 Placeholder 모드를 사용합니다. 실제 CLIP 모델을 사용하려면 환경변수 ENABLE_CLIP=1을 설정하세요.")
            vec = compute_embedding_placeholder(image)
        logger.info(f"[벡터화 성공] 벡터 차원: {len(vec)}, 벡터 타입: {type(vec).__name__}")
    except Exception as e:
        logger.error(f"[벡터화 실패] {e}")
        raise HTTPException(status_code=500, detail="Embedding failure")

    vec_text = vec_to_sql_literal(vec)
    logger.info(f"[검색] 벡터를 SQL 리터럴로 변환 완료 (길이: {len(vec_text)} 문자)")

    try:
        tgt_conn = psycopg2.connect(host=TGT_DB_HOST, port=TGT_DB_PORT, dbname=TGT_DB_NAME, user=TGT_DB_USER, password=TGT_DB_PASS)
        candidate_limit = min(max(top_k * 10, top_k), 500)
        logger.info(f"[검색] 타겟 DB에서 후보 검색 시작 (candidate_limit: {candidate_limit})")
        with tgt_conn.cursor() as cur:
            sql = "SELECT product_id, vector <=> %s::vector AS distance FROM product_vectors ORDER BY distance ASC LIMIT %s"
            cur.execute(sql, (vec_text, candidate_limit))
            rows = cur.fetchall()
        logger.info(f"[검색] 타겟 DB에서 {len(rows)}개 후보 발견")
    except Exception as e:
        logger.error(f"[검색] 타겟 DB 쿼리 실패: {e}")
        raise HTTPException(status_code=500, detail="Target DB query failed")
    finally:
        try:
            tgt_conn.close()
        except Exception:
            pass

    results = []
    if not rows:
        logger.warning("[검색] 검색 결과가 없습니다")
        return results

    try:
        src_conn = psycopg2.connect(host=SRC_DB_HOST, port=SRC_DB_PORT, dbname=SRC_DB_NAME, user=SRC_DB_USER, password=SRC_DB_PASS)
        logger.info(f"[검색] 소스 DB에서 제품 정보 조회 시작")
        with src_conn.cursor() as cur:
            rank = 1
            for pid, dist in rows:
                try:
                    cur.execute("SELECT id, image_url, COALESCE(name, '') FROM product WHERE id = %s AND is_active = true", (pid,))
                    rec = cur.fetchone()
                except Exception as e:
                    logger.warning(f"[검색] 제품 {pid} 조회 실패: {e}")
                    rec = None

                if not rec:
                    logger.debug(f"[검색] 제품 {pid}는 활성화되지 않았거나 존재하지 않음")
                    continue

                image_url = rec[1] if len(rec) > 1 else None
                try:
                    d = float(dist)
                    # 코사인 거리(<=>)를 코사인 유사도로 변환: 유사도 = 1 - 거리
                    # 코사인 거리는 0(완전히 유사) ~ 2(완전히 반대) 범위
                    # 코사인 유사도는 -1 ~ 1 범위이므로, 0~1 범위로 정규화
                    cos_sim = 1.0 - d  # 코사인 거리를 코사인 유사도로 변환
                    cos_sim = max(-1.0, min(1.0, cos_sim))  # 범위 제한
                    sim01 = (cos_sim + 1.0) / 2.0  # -1~1을 0~1로 정규화
                    logger.info(f"[검색 결과] Rank {rank}: 제품 {pid} - 거리: {d:.6f}, 코사인 유사도: {cos_sim:.6f}, 최종 점수: {sim01:.6f}")
                except Exception as e:
                    logger.warning(f"[검색] 제품 {pid} 점수 계산 실패: {e}")
                    sim01 = 0.0

                results.append({"product_id": str(pid), "image_url": image_url, "score": float(sim01), "rank": rank})
                rank += 1
                if rank > top_k:
                    break
        
        # 최종 결과 요약
        if results:
            scores = [r["score"] for r in results]
            logger.info(f"[검색 완료] 총 {len(results)}개 결과 반환 - 최고 점수: {max(scores):.6f}, 최저 점수: {min(scores):.6f}, 평균 점수: {sum(scores)/len(scores):.6f}")
        else:
            logger.warning("[검색 완료] 결과가 없습니다")
    except Exception as e:
        for pid, dist in rows:
            try:
                d = float(dist)
                # 코사인 거리(<=>)를 코사인 유사도로 변환: 유사도 = 1 - 거리
                cos_sim = 1.0 - d
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
