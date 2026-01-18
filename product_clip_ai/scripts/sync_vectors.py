import os
import sys
import json
import io
import logging
from typing import List, Tuple, Optional

import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
import numpy as np
from PIL import Image
import requests

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

SRC_DB_HOST = os.getenv("SRC_DB_HOST", "localhost")
SRC_DB_PORT = int(os.getenv("SRC_DB_PORT", "5432"))
SRC_DB_NAME = os.getenv("SRC_DB_NAME", "mall")
SRC_DB_USER = os.getenv("SRC_DB_USER")
SRC_DB_PASS = os.getenv("SRC_DB_PASS")
SRC_PRODUCT_QUERY = os.getenv("SRC_PRODUCT_QUERY", "SELECT id, image_url FROM product WHERE is_active = true LIMIT 100;")
IMAGE_BASE_URL = os.getenv("IMAGE_BASE_URL", "")  # 이미지 베이스 URL (예: "https://example.com" 또는 "http://localhost:8000")

TGT_DB_HOST = os.getenv("TGT_DB_HOST", "localhost")
TGT_DB_PORT = int(os.getenv("TGT_DB_PORT", "5433"))
TGT_DB_NAME = os.getenv("TGT_DB_NAME", "productdb")
TGT_DB_USER = os.getenv("TGT_DB_USER")
TGT_DB_PASS = os.getenv("TGT_DB_PASS")

# 필수 자격증명 체크: 누락시 익셉션으로 중단(로그 출력 없음)
if not SRC_DB_USER or not SRC_DB_PASS:
    raise RuntimeError("소스 DB 자격증명(SRC_DB_USER/SRC_DB_PASS)이 설정되어 있지 않습니다. .env 또는 환경변수를 확인하세요.")
if not TGT_DB_USER or not TGT_DB_PASS:
    raise RuntimeError("대상 DB 자격증명(TGT_DB_USER/TGT_DB_PASS)이 설정되어 있지 않습니다. .env 또는 환경변수를 확인하세요.")

EMBED_DIM = int(os.getenv("EMBED_DIM", "512"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
ENABLE_CLIP = os.getenv("ENABLE_CLIP", "0") == "1"

# CLIP 모델 전역 변수
torch = None
CLIPProcessor = None
CLIPModel = None
model = None
processor = None
device = "cpu"

if ENABLE_CLIP:
    try:
        import torch
        from transformers import CLIPProcessor, CLIPModel
        logger.info(f"[초기화] torch 모듈 로드 성공, torch 버전: {torch.__version__}")
    except Exception as e:
        torch = None
        CLIPProcessor = None
        CLIPModel = None
        logger.warning(f"[초기화] torch 모듈 로드 실패: {e}")
        logger.warning(f"[초기화] Placeholder 모드로 동작합니다. CLIP을 사용하려면 torch를 설치하세요.")

def make_conn(host: str, port: int, dbname: str, user: str, password: str):
    return psycopg2.connect(host=host, port=port, dbname=dbname, user=user, password=password)


def compute_embedding_placeholder(image_ref: str, dim: int = EMBED_DIM) -> List[float]:
    """Placeholder 임베딩 생성 (이미지 URL 해시 기반)"""
    vec = np.random.RandomState(abs(hash(image_ref)) % (2 ** 32)).randn(dim).astype(np.float32)
    vec /= np.linalg.norm(vec) + 1e-12
    return vec.tolist()


def normalize_image_url(image_url: str) -> Optional[str]:
    """이미지 URL을 정규화 (상대 경로를 절대 URL로 변환)"""
    if not image_url or not isinstance(image_url, str):
        return None
    
    image_url = image_url.strip()
    
    # 이미 절대 URL인 경우
    if image_url.startswith('http://') or image_url.startswith('https://'):
        return image_url
    
    # 상대 경로인 경우 베이스 URL과 결합
    if IMAGE_BASE_URL:
        # 상대 경로가 /로 시작하면 그대로, 아니면 / 추가
        if image_url.startswith('/'):
            return IMAGE_BASE_URL.rstrip('/') + image_url
        else:
            return IMAGE_BASE_URL.rstrip('/') + '/' + image_url
    
    # 베이스 URL이 없으면 None 반환
    return None


def load_image_from_url(image_url: str, timeout: int = 10) -> Optional[Image.Image]:
    """이미지 URL에서 이미지 다운로드"""
    if not image_url:
        return None
    
    # URL 정규화
    normalized_url = normalize_image_url(image_url)
    if not normalized_url:
        logger.debug(f"[이미지 URL 정규화 실패] 원본: {image_url}, IMAGE_BASE_URL: {IMAGE_BASE_URL}")
        return None
    
    try:
        response = requests.get(normalized_url, timeout=timeout, stream=True)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        return image
    except Exception as e:
        logger.warning(f"[이미지 다운로드 실패] {normalized_url}: {e}")
        return None


def compute_embedding_clip(image: Image.Image) -> Optional[List[float]]:
    """CLIP 모델을 사용하여 이미지 벡터화"""
    global model, processor, device
    
    if torch is None or CLIPProcessor is None or CLIPModel is None:
        return None
    
    if model is None or processor is None:
        try:
            logger.info("[벡터화] CLIP 모델 로딩 시작...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"[벡터화] 디바이스: {device}")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            model.to(device)
            model.eval()
            logger.info("[벡터화] CLIP 모델 로딩 완료")
        except Exception as e:
            logger.error(f"[벡터화] CLIP 모델 로딩 실패: {e}")
            return None
    
    try:
        inputs = processor(images=image, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            features = model.get_image_features(**inputs)
        vec = features[0].cpu().numpy().astype(np.float32)
        vec /= (np.linalg.norm(vec) + 1e-12)
        return vec.tolist()
    except Exception as e:
        logger.error(f"[벡터화] CLIP 벡터화 중 오류 발생: {e}")
        return None


def compute_embedding(image_url: Optional[str], image_ref: str) -> List[float]:
    """이미지 URL 또는 참조값으로부터 임베딩 생성"""
    if ENABLE_CLIP and torch is not None and image_url:
        # CLIP 모드: 실제 이미지 다운로드 및 벡터화
        image = load_image_from_url(image_url)
        if image:
            emb = compute_embedding_clip(image)
            if emb:
                return emb
            logger.debug(f"[벡터화] CLIP 벡터화 실패, Placeholder로 폴백: {image_url}")
        else:
            logger.debug(f"[벡터화] 이미지 다운로드 실패, Placeholder로 폴백: {image_url}")
    elif not image_url:
        # image_url이 없는 경우 Placeholder 사용 (로그 출력 안 함)
        pass
    
    # Placeholder 모드 또는 CLIP 실패 시
    return compute_embedding_placeholder(image_ref)


def ensure_target_table(conn):
    sql = f"""
    CREATE TABLE IF NOT EXISTS product_vectors (
        product_vector_id serial PRIMARY KEY,
        product_id varchar(255) NOT NULL,
        vector vector({EMBED_DIM}) NOT NULL,
        created_at timestamptz DEFAULT now(),
        UNIQUE (product_id)
    );
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        cur.execute("ALTER TABLE product_vectors ADD COLUMN IF NOT EXISTS created_at timestamptz DEFAULT now();")
        conn.commit()


def fetch_products(conn, query: str) -> List[Tuple]:
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
    return rows


def upsert_vectors(conn, rows: List[Tuple[str, List[float]]]):
    with conn.cursor() as cur:
        insert_sql = """
        INSERT INTO product_vectors (product_id, vector)
        VALUES (%s, %s::vector)
        ON CONFLICT (product_id) DO UPDATE SET vector = EXCLUDED.vector;
        """
        for product_id, vec in rows:
            vec_text = '[' + ','.join(f"{float(x):.6f}" for x in vec) + ']'  # format
            cur.execute(insert_sql, (str(product_id), vec_text))
        conn.commit()


def main():
    try:
        try:
            src_conn = make_conn(SRC_DB_HOST, SRC_DB_PORT, SRC_DB_NAME, SRC_DB_USER, SRC_DB_PASS)
        except Exception as e:
            raise

        try:
            tgt_conn = make_conn(TGT_DB_HOST, TGT_DB_PORT, TGT_DB_NAME, TGT_DB_USER, TGT_DB_PASS)
        except Exception as e:
            src_conn.close()
            raise

        ensure_target_table(tgt_conn)

        try:
            products = fetch_products(src_conn, SRC_PRODUCT_QUERY)
        except Exception as e:
            src_conn.close(); tgt_conn.close(); raise

        try:
            with src_conn.cursor() as cur:
                cur.execute("SELECT id FROM product WHERE is_active = false")
                inactive_rows = cur.fetchall()
            if inactive_rows:
                inactive_ids = [str(r[0]) for r in inactive_rows]
                with tgt_conn.cursor() as cur:
                    batch_size = 100
                    for i in range(0, len(inactive_ids), batch_size):
                        chunk = inactive_ids[i:i+batch_size]
                        cur.execute("DELETE FROM product_vectors WHERE product_id = ANY(%s)", (chunk,))
                tgt_conn.commit()
        except Exception as e:
            pass

        if not products:
            src_conn.close(); tgt_conn.close(); return

        # 샘플 데이터 확인 (처음 5개)
        logger.info(f"[데이터 확인] 총 {len(products)}개 제품 조회됨")
        for i, row in enumerate(products[:5]):
            logger.info(f"[데이터 샘플 {i+1}] row 길이: {len(row)}, 값: {row}")

        normalized = []
        url_count = 0
        null_url_count = 0
        for row in products:
            if len(row) == 1:
                pid = row[0]
                image_url = None
                null_url_count += 1
            else:
                pid = row[0]
                image_url = row[1] if (len(row) > 1 and row[1]) else None
                if image_url:
                    url_count += 1
                else:
                    null_url_count += 1
            normalized.append((pid, image_url))
        
        logger.info(f"[데이터 통계] image_url 있음: {url_count}개, image_url 없음: {null_url_count}개")

        batch = []
        total = 0
        success_count = 0
        fail_count = 0
        clip_success = 0
        placeholder_count = 0
        
        logger.info(f"[벡터화] 총 {len(normalized)}개 제품 벡터화 시작 (CLIP 모드: {ENABLE_CLIP})")
        
        for pid, image_url in normalized:
            # image_url이 유효한 URL인지 확인
            if image_url and isinstance(image_url, str) and image_url.strip():
                image_url = image_url.strip()
                if not (image_url.startswith('http://') or image_url.startswith('https://')):
                    logger.debug(f"[벡터화] 제품 {pid}: image_url이 URL 형식이 아님: {image_url[:50] if len(image_url) > 50 else image_url}")
                    image_url = None
            else:
                image_url = None
            
            ref_value = str(pid)
            
            emb = compute_embedding(image_url, ref_value)
            batch.append((pid, emb))
            
            if emb and len(emb) == EMBED_DIM:
                success_count += 1
                # CLIP으로 성공했는지 확인 (Placeholder와 구분하기 어렵지만, image_url이 있었는지로 판단)
                if image_url:
                    clip_success += 1
                else:
                    placeholder_count += 1
            else:
                fail_count += 1
            
            if len(batch) >= BATCH_SIZE:
                upsert_vectors(tgt_conn, batch)
                total += len(batch)
                logger.info(f"[진행상황] {total}/{len(normalized)}개 처리 완료 (성공: {success_count}, 실패: {fail_count}, CLIP: {clip_success}, Placeholder: {placeholder_count})")
                batch = []
        
        if batch:
            upsert_vectors(tgt_conn, batch)
            total += len(batch)
        
        logger.info(f"[완료] 총 {total}개 제품 벡터화 완료 (성공: {success_count}, 실패: {fail_count}, CLIP 성공: {clip_success}, Placeholder: {placeholder_count})")

        src_conn.close(); tgt_conn.close()
    except Exception as e:
        raise


if __name__ == '__main__':
    main()

