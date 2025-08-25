"""
데이터 모델 정의 (Pydantic Schemas)
"""
from pydantic import BaseModel
from typing import Optional


class AvatarGenerationTask(BaseModel):
    """아바타 생성 작업 모델"""
    user_text: str        # 사용자 입력
    ai_response: str      # AI 응답
    session_id: str       # 세션 ID


class AvatarGenerationResult(BaseModel):
    """아바타 생성 결과 모델"""
    video_url: str                    # 비디오 파일 경로
    audio_url: Optional[str] = None   # 오디오 파일 경로
    duration: float                   # 재생 시간 (초)
    text: str                         # 응답 텍스트
