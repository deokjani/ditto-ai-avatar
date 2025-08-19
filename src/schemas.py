"""
데이터 모델
"""
from pydantic import BaseModel
from typing import Optional, Literal

class WSMessage(BaseModel):
    type: Literal["send_message", "video_ended", "get_status"]
    text: Optional[str] = None
    video_url: Optional[str] = None

class VideoTask(BaseModel):
    user_text: str
    ai_response: str
    session_id: str

class VideoResult(BaseModel):
    video_url: str
    audio_url: Optional[str] = None
    duration: float
    text: str
