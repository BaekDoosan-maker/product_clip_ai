# Dockerfile
FROM python:3.10-slim

RUN python -m pip install --upgrade pip

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필요한 라이브러리 설치
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    build-essential \
    libsndfile1 \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치: 먼저 CPU 전용 PyTorch 휠을 설치해 큰 CUDA 휠 다운로드를 방지
COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir "torch" --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# 애플리케이션 파일 복사
COPY test_clip_similarity.py .

# 데이터 저장용 디렉토리 생성
RUN mkdir -p /app/clip_test_data/random_images \
             /app/clip_test_data/style_images \
             /app/clip_test_data/results

# 볼륨 마운트 포인트 선언
VOLUME ["/app/clip_test_data"]

# 기본 실행 명령
CMD ["python", "test_clip_similarity.py"]
