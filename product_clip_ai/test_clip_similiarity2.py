import os
import sys
import argparse
try:
    import torch
except Exception as e:
    torch = None
try:
    import clip
except Exception as e:
    clip = None
from PIL import Image
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()

# 1. 환경 설정
DEVICE = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B/32"
IMAGE_DIR = os.getenv("IMAGE_SOURCE_DIR")
INSERT_QUERY = os.getenv("SQL_INSERT_PRODUCT_VECTOR")

def run_bulk_insert(dry_run: bool = False):
    if clip is None or torch is None:
        print("필수 패키지(pytorch/clip)가 설치되어 있지 않거나 import에 실패했습니다.")
        print("설치 예: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # or cpu wheels")
        print("그리고 OpenAI CLIP: pip install git+https://github.com/openai/CLIP.git")
        return

    try:
        model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
    except Exception as e:
        print(f"CLIP 모델을 불러오는 중 오류 발생: {e}")
        return

    data_to_insert = []

    if not IMAGE_DIR:
        print("환경변수 IMAGE_SOURCE_DIR가 설정되어 있지 않습니다.")
        return

    if not os.path.exists(IMAGE_DIR):
        print(f"경로를 찾을 수 없습니다: {IMAGE_DIR}")
        return

    # 3. 폴더 내 이미지들 하나씩 분석
    for filename in os.listdir(IMAGE_DIR):
        if not filename.startswith("img_"):
            continue

        try:
            img_path = os.path.join(IMAGE_DIR, filename)
            # 파일명에서 ID 추출 (img_1.png -> 1)
            product_id = filename.replace("img_", "").split(".")[0]

            # [핵심 분석 코드] 이미지를 읽어서 512차원 벡터로 변환
            image = preprocess(Image.open(img_path)).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                vector = model.encode_image(image).cpu().numpy().flatten().tolist()

            # 리스트에 추가
            data_to_insert.append((product_id, vector))

        except Exception as e:
            print(f"이미지 처리 실패({filename}): {e}")
            continue # 실패한 이미지는 그냥 넘어감

    # 4. DB에 한 번에 쏟아붓기 (Bulk Insert)
    if not data_to_insert:
        print("삽입할 데이터가 없습니다.")
        return

    if dry_run:
        print(f"Dry run: 준비된 레코드 수 = {len(data_to_insert)}")
        # 샘플 출력
        for i, item in enumerate(data_to_insert[:5]):
            print(f"샘플 {i+1}: id={item[0]} vector_len={len(item[1])}")
        # SQL 템플릿 확인
        print(f"사용될 INSERT 템플릿(SQL_INSERT_PRODUCT_VECTOR) = {INSERT_QUERY}")
        return

    # DB 접속 (환경변수 이름 호환성 처리)
    db_password = os.getenv("DB_PASS") or os.getenv("DB_PASSWORD")
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=db_password,
            port=os.getenv("DB_PORT") or 5432
        )
        cur = conn.cursor()

        if not INSERT_QUERY:
            print("환경변수 SQL_INSERT_PRODUCT_VECTOR가 설정되어 있지 않습니다. 예: INSERT INTO product (id, vector) VALUES %s")
            cur.close()
            conn.close()
            return

        # execute_values로 다수 행 인서트
        execute_values(cur, INSERT_QUERY, data_to_insert)

        conn.commit()
        cur.close()
        conn.close()
        print(f"DB에 {len(data_to_insert)}개 레코드를 저장했습니다.")
    except Exception as e:
        print(f"DB 저장 실패: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP 이미지 벡터를 생성해 DB에 저장합니다.")
    parser.add_argument("--dry-run", action="store_true", help="DB에 쓰지 않고 이미지 인코딩까지만 실행")
    args = parser.parse_args()

    run_bulk_insert(dry_run=args.dry_run)
