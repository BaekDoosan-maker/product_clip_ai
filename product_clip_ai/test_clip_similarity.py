"""
CLIP 이미지 유사도 테스트 스크립트
- Unsplash API로 랜덤 이미지 400장 다운로드
- 사용자 제공 스타일 이미지와 유사도 비교
- CPU에서 실행 가능
"""
import os
import requests
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple
import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import json


class CLIPSimilarityTester:
    def __init__(self, model_name: str = None):
        """
        CLIP 유사도 테스터 초기화

        Args:
            model_name: 사용할 CLIP 모델 (기본: ViT-B/32, 512차원, 경량-빠름). 환경변수 MODEL_NAME로도 지정 가능
        """
        # 우선순위: 인자로 받은 model_name -> 환경변수 MODEL_NAME -> 기본값
        default_model = "openai/clip-vit-base-patch32"
        model_name = model_name or os.environ.get("MODEL_NAME") or default_model

        print(f"Loading CLIP model: {model_name}")
        self.device = "cpu"  # CPU 사용
        try:
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model.eval()
        except Exception as e:
            print(f"Failed to load model '{model_name}': {e}")
            raise

        # 저장 경로 설정
        self.base_dir = Path("clip_test_data")
        self.random_images_dir = self.base_dir / "random_images"
        self.style_images_dir = self.base_dir / "style_images"
        self.results_dir = self.base_dir / "results"

        # 디렉토리 생성
        for dir_path in [self.random_images_dir, self.style_images_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def download_unsplash_images(self, count: int = 400):
        """
        랜덤 이미지 다운로드 (Picsum Photos 사용)

        Args:
            count: 다운로드할 이미지 수
        """
        print(f"\nDownloading {count} random images from Picsum Photos...")

        # Picsum Photos API (무료, 안정적)
        base_url = "https://picsum.photos/800/600"

        downloaded = 0
        failed = 0
        for i in tqdm(range(count), desc="Downloading"):
            image_path = self.random_images_dir / f"random_{i:04d}.jpg"

            # 이미 다운로드된 경우 스킵
            if image_path.exists():
                downloaded += 1
                continue

            try:
                # 랜덤 이미지 다운로드 (매번 다른 이미지)
                # random parameter를 추가해서 매번 다른 이미지 받기
                response = requests.get(f"{base_url}?random={i}", timeout=15, allow_redirects=True)
                if response.status_code == 200:
                    with open(image_path, 'wb') as f:
                        f.write(response.content)
                    downloaded += 1
                else:
                    failed += 1
                    if failed < 5:  # 처음 5개 에러만 출력
                        print(f"\nFailed to download image {i}: HTTP {response.status_code}")
            except Exception as e:
                failed += 1
                if failed < 5:
                    print(f"\nError downloading image {i}: {e}")

        print(f"Successfully downloaded {downloaded}/{count} images")
        return downloaded

    def encode_image(self, image_path: str) -> np.ndarray:
        """
        이미지를 CLIP 벡터로 인코딩

        Args:
            image_path: 이미지 파일 경로

        Returns:
            512차원 벡터 (numpy array)
        """
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                # 정규화
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            return image_features.cpu().numpy().flatten()
        except Exception as e:
            print(f"Error encoding {image_path}: {e}")
            return None

    def compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        코사인 유사도 계산

        Args:
            vec1, vec2: 정규화된 벡터

        Returns:
            유사도 (0~1)
        """
        return float(np.dot(vec1, vec2))

    def vectorize_all_images(self) -> Tuple[dict, np.ndarray]:
        """
        모든 랜덤 이미지 벡터화

        Returns:
            (이미지 경로 매핑, 벡터 행렬)
        """
        print("\nVectorizing random images with CLIP...")

        image_files = sorted(self.random_images_dir.glob("*.jpg"))
        vectors = []
        image_paths = []

        for img_path in tqdm(image_files, desc="Encoding"):
            vec = self.encode_image(str(img_path))
            if vec is not None:
                vectors.append(vec)
                image_paths.append(str(img_path))

        vectors_matrix = np.vstack(vectors)

        print(f"Encoded {len(vectors)} images")
        print(f"Vector shape: {vectors_matrix.shape}")

        return {"paths": image_paths}, vectors_matrix

    def find_similar_images(
        self,
        style_image_path: str,
        random_vectors: np.ndarray,
        image_metadata: dict,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        스타일 이미지와 유사한 이미지 찾기

        Args:
            style_image_path: 스타일 이미지 경로
            random_vectors: 랜덤 이미지 벡터 행렬
            image_metadata: 이미지 경로 매핑
            top_k: 상위 K개 결과

        Returns:
            [(이미지 경로, 유사도), ...] 리스트
        """
        print(f"\nFinding similar images to: {style_image_path}")

        # 스타일 이미지 벡터화
        style_vector = self.encode_image(style_image_path)
        if style_vector is None:
            print("Failed to encode style image")
            return []

        # 모든 이미지와 유사도 계산
        similarities = []
        for i, random_vec in enumerate(random_vectors):
            sim = self.compute_similarity(style_vector, random_vec)
            similarities.append((image_metadata["paths"][i], sim))

        # 유사도 기준 내림차순 정렬
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Top-K 결과
        top_results = similarities[:top_k]

        print(f"\nTop-{top_k} Similar Images:")
        for rank, (img_path, score) in enumerate(top_results, 1):
            filename = Path(img_path).name
            print(f"  {rank}. {filename} - Similarity: {score:.4f}")

        return top_results

    def save_results(self, results: List[Tuple[str, float]], style_image_path: str):
        """
        결과를 JSON 파일로 저장

        Args:
            results: 유사도 결과 리스트
            style_image_path: 스타일 이미지 경로
        """
        output_data = {
            "style_image": str(style_image_path),
            "timestamp": str(Path(style_image_path).stat().st_mtime),
            "results": [
                {
                    "rank": i + 1,
                    "image_path": img_path,
                    "similarity_score": float(score)
                }
                for i, (img_path, score) in enumerate(results)
            ]
        }

        output_file = self.results_dir / "similarity_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {output_file}")

    def visualize_results(self, results: List[Tuple[str, float]], style_image_path: str):
        """
        결과를 HTML로 시각화

        Args:
            results: 유사도 결과 리스트
            style_image_path: 스타일 이미지 경로
        """
        from urllib.parse import quote

        style_filename = Path(style_image_path).name
        style_filename_encoded = quote(style_filename)

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>CLIP Similarity Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
        h1 {{ color: #333; }}
        .style-image {{ margin: 20px 0; }}
        .style-image img {{ max-width: 400px; border: 3px solid #007bff; border-radius: 4px; }}
        .results {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 20px; margin-top: 30px; }}
        .result-item {{ background: #f9f9f9; padding: 10px; border-radius: 4px; text-align: center; }}
        .result-item img {{ width: 100%; height: 200px; object-fit: cover; border-radius: 4px; }}
        .rank {{ font-weight: bold; color: #007bff; font-size: 18px; }}
        .score {{ color: #666; margin-top: 5px; }}
        .score.high {{ color: #28a745; font-weight: bold; }}
        .score.medium {{ color: #ffc107; }}
        .score.low {{ color: #dc3545; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>CLIP Image Similarity Test Results</h1>

        <div class="style-image">
            <h2>Style Reference Image</h2>
            <p><strong>{style_filename}</strong></p>
            <img src="../style_images/{style_filename_encoded}" alt="Style Image">
        </div>

        <h2>Top Similar Images</h2>
        <div class="results">
"""

        for rank, (img_path, score) in enumerate(results, 1):
            img_filename = Path(img_path).name
            relative_path = f"../random_images/{img_filename}"

            # 유사도에 따라 색상 클래스 지정
            if score >= 0.7:
                score_class = "high"
            elif score >= 0.5:
                score_class = "medium"
            else:
                score_class = "low"

            html_content += f"""            <div class="result-item">
                <div class="rank">#{rank}</div>
                <img src="{relative_path}" alt="Result {rank}">
                <div class="score {score_class}">Similarity: {score:.4f}</div>
            </div>
"""

        html_content += """        </div>
    </div>
</body>
</html>
"""

        output_file = self.results_dir / "results.html"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"Visualization saved to: {output_file}")


def main():
    """메인 실행 함수"""
    print("="*60)
    print("CLIP Image Similarity Tester")
    print("="*60)

    # CLIP 테스터 초기화
    tester = CLIPSimilarityTester()

    # 1. 랜덤 이미지 다운로드 (이미 있으면 스킵)
    tester.download_unsplash_images(count=400)

    # 2. 스타일 이미지 확인
    style_images = list(tester.style_images_dir.glob("*.jpg")) + \
                   list(tester.style_images_dir.glob("*.png"))

    if not style_images:
        print("\n" + "="*60)
        print("⚠️  스타일 이미지를 추가해주세요!")
        print("="*60)
        print(f"\n다음 폴더에 원하는 스타일의 이미지를 넣어주세요:")
        print(f"  {tester.style_images_dir.absolute()}")
        print(f"\n지원 형식: .jpg, .png")
        print("\n이미지를 추가한 후 다시 실행하세요.")
        return

    style_image_path = str(style_images[0])
    print(f"\nUsing style image: {Path(style_image_path).name}")

    # 3. 랜덤 이미지 벡터화
    metadata, vectors = tester.vectorize_all_images()

    # 4. 유사도 검색
    results = tester.find_similar_images(
        style_image_path=style_image_path,
        random_vectors=vectors,
        image_metadata=metadata,
        top_k=30
    )

    # 5. 결과 저장
    tester.save_results(results, style_image_path)
    tester.visualize_results(results, style_image_path)

    print("\n" + "="*60)
    print("✅ Test completed successfully!")
    print("="*60)
    print(f"\nResults:")
    print(f"  - JSON: {tester.results_dir / 'similarity_results.json'}")
    print(f"  - HTML: {tester.results_dir / 'results.html'}")
    print(f"\nHTML 파일을 브라우저로 열어서 결과를 확인하세요!")


if __name__ == "__main__":
    main()
