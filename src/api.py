"""
API 엔드포인트
"""
import uuid
from pathlib import Path
from fastapi import WebSocket, WebSocketDisconnect, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from schemas import VideoTask
from services import chat_service, video_service

# 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent
TEMPLATES_DIR = PROJECT_ROOT / "templates"
BASE_VIDEO = PROJECT_ROOT / "example" / "base.mp4"
TEMP_DIR = Path.home() / ".cache" / "ditto_temp"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# REST API
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

async def health_check():
    return JSONResponse({"status": "healthy"})

async def get_default_video():
    return FileResponse(str(BASE_VIDEO), media_type='video/mp4')

async def get_video(filename: str):
    video_path = TEMP_DIR / "videos" / filename
    if not video_path.exists():
        return JSONResponse({"error": "Video not found"}, status_code=404)
    return FileResponse(str(video_path), media_type='video/mp4')

async def get_audio(filename: str):
    audio_path = TEMP_DIR / "audio" / filename
    if not audio_path.exists():
        return JSONResponse({"error": "Audio not found"}, status_code=404)
    return FileResponse(str(audio_path), media_type='audio/wav')

# 채팅 API (REST)
async def chat(request: Request):
    """채팅 REST API - 즉시 응답 반환"""
    data = await request.json()
    user_text = data.get('text', '')
    session_id = data.get('session_id', str(uuid.uuid4()))
    
    if not user_text:
        return JSONResponse({"error": "No text provided"}, status_code=400)
    
    # AI 응답 생성 (invoke 방식)
    ai_response = await chat_service.get_response(user_text, session_id)
    
    # 비디오 작업 큐에 추가
    if ai_response:
        await video_service.add_task(VideoTask(
            user_text=user_text,
            ai_response=ai_response,
            session_id=session_id
        ))
    
    return JSONResponse({
        "response": ai_response,
        "session_id": session_id
    })

# WebSocket - 비디오 스트리밍 전용
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket - 비디오 준비 알림 전용"""
    await websocket.accept()
    video_service.add_connection(websocket)
    
    try:
        # 연결 유지 (비디오 알림만 받음)
        while True:
            # 클라이언트의 ping/pong 메시지 처리
            await websocket.receive_text()
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        # WebSocket error (조용히 처리)
        pass
    finally:
        video_service.remove_connection(websocket)
