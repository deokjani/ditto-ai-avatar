"""
환경 변수 설정
"""
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# API 키
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TTS_API_KEY = os.getenv("GOOGLE_TTS_API_KEY")

# Redis 설정
USE_REDIS = False
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", 6379)
REDIS_DB = os.getenv("REDIS_DB", 0)
REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
