프로젝트 개요

이 저장소는 이미지 기반 제품 유사도 검색(벡터 검색) API와 벡터 동기화 스크립트를 포함합니다.
- `api.py`: 업로드된 이미지로 임베딩을 생성하고 DB에 저장된 제품 벡터와 유사도를 계산하여 결과를 반환합니다.
- `scripts/sync_vectors.py`: 소스 DB의 활성 제품만 읽어 벡터를 생성(또는 삭제된 제품 벡터 제거)해 타깃 DB에 업서트합니다.

핵심 동작 원리 (간략)

1. 임베딩 생성
- 기본적으로 placeholder 임베딩(결정론적 난수)을 사용합니다.
- 실제 OpenAI CLIP 모델을 사용하려면 환경변수 `ENABLE_CLIP=1`을 설정하면, 최초 해당 엔드포인트 호출 시 모델을 lazy-load(지연 로드)해서 임베딩을 생성합니다. 모델 로드 실패 시 자동으로 placeholder로 폴백합니다.

2. 검색 순서
- 클라이언트가 이미지 업로드 -> 임베딩 생성 -> 타깃 DB(`product_vectors`)에서 거리(distance) 기준 상위 후보를 가져옴 -> 소스 DB에서 `is_active = true`인 제품만 필터링 후 최종 top-k 반환.
- 비활성(`is_active = false`) 제품의 벡터는 `scripts/sync_vectors.py`가 주기적으로 제거하도록 되어 있어 검색 결과에 나타나지 않습니다.

필수 환경변수

(앱과 스크립트 모두 동일한 환경변수명을 사용합니다)
- SRC_DB_HOST (예: localhost)
- SRC_DB_PORT (예: 5432)
- SRC_DB_NAME (예: mall)
- SRC_DB_USER
- SRC_DB_PASS
- TGT_DB_HOST (벡터 저장 DB; 기본은 localhost)
- TGT_DB_PORT
- TGT_DB_NAME
- TGT_DB_USER
- TGT_DB_PASS
- ENABLE_CLIP (선택, 기본 0) — 1로 설정하면 CLIP 사용 시도
- EMBED_DIM (선택, 기본 512)
- BATCH_SIZE (선택, sync 스크립트 배치 크기)
- DEFAULT_TOP_K (선택, API 기본 top-k)


로컬 실행 방법 (PowerShell 예)

1) 환경변수 설정(.env 사용 권장)
```powershell
# 환경변수 설정
$env:SRC_DB_USER=''; $env:SRC_DB_PASS=''; $env:SRC_DB_HOST=''; $env:SRC_DB_PORT='';
$env:TGT_DB_USER=''; $env:TGT_DB_PASS=''; $env:TGT_DB_HOST=''; $env:TGT_DB_PORT='';
```

2) API 서버 실행 (uvicorn)
```powershell
python -m uvicorn api:app --host 0.0.0.0 --port 9000
```
- `ENABLE_CLIP=1`로 설정하면 첫 요청에서 모델 다운로드가 시도됩니다. 네트워크 불안정 시 타임아웃이 발생할 수 있으므로 사전 캐시를 권장합니다.



벡터 동기화 스크립트 사용

- DB에서 활성 제품의 이미지 URL을 읽어 벡터를 생성해 타깃 DB에 저장합니다.
- 실행:
```powershell
python scripts/sync_vectors.py
```
- 주기적으로(크론, CI 파이프라인, 또는 도커 스케줄러 등) 호출해 최신 상태를 유지하세요.

도커 컴포즈

- `docker-compose.yml`은 환경변수(또는 .env)에서 DB 정보를 읽도록 구성되어 있습니다.
- 실행:
```powershell
# .env를 준비한 뒤
docker-compose up -d
```

API 엔드포인트

- GET /health
  - 응답: 서비스 준비 여부와 모델 사용 가능성(MODEL_AVAILABLE 플래그)
  
- POST /search_similar (multipart/form-data, file)
  - 파라미터: file(이미지), top_k (선택)
  - 반환: product_id, image_url, score, rank 배열





