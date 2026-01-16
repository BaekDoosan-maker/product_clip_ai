import requests
import os

API_URL = os.environ.get('API_URL', 'http://localhost:8080')

# 로컬 테스트용 이미지 경로
TEST_IMAGE = 'clip_test_data/style_images/test_upload.jpg'

if not os.path.exists(TEST_IMAGE):
    print('테스트 이미지가 없습니다. clip_test_data/style_images/test_upload.jpg 파일을 넣어주세요.')
    exit(1)

files = {'file': open(TEST_IMAGE, 'rb')}
params = {'top_k': 5}

resp = requests.post(f"{API_URL}/find_similar_upload", files=files, params=params)
print('Status:', resp.status_code)
print(resp.json())

