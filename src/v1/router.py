"""
API 라우터 - 모든 엔드포인트 정의
"""
import uuid
from pathlib import Path
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from v1.schemas import AvatarGenerationTask
from v1.services import chat_service, avatar_generation_service

router = APIRouter()

# 경로 설정
ROOT = Path(__file__).parent.parent.parent  # v1 -> src -> project_root
BASE_VIDEO = ROOT / "example" / "base.mp4"
TEMP_DIR = Path.home() / ".cache" / "ditto_temp"
templates = Jinja2Templates(directory=str(ROOT / "templates"))

# ========== 페이지 라우트 ==========

@router.get("/")
async def index(request: Request):
    """메인 페이지 렌더링"""
    return templates.TemplateResponse("index.html", {"request": request})

@router.get("/health")
async def health_check():
    """서버 상태 확인"""
    return JSONResponse({"status": "healthy"})

# ========== 미디어 라우트 ==========

@router.get("/default_video")
async def default_video():
    """기본 아바타 비디오 제공"""
    return FileResponse(str(BASE_VIDEO), media_type='video/mp4')

@router.get("/video/{filename}")
async def get_video(filename: str):
    """생성된 비디오 파일 제공"""
    path = TEMP_DIR / "videos" / filename
    if not path.exists():
        return JSONResponse({"error": "Video not found"}, status_code=404)
    return FileResponse(str(path), media_type='video/mp4')

@router.get("/audio/{filename}")
async def get_audio(filename: str):
    """생성된 오디오 파일 제공"""
    path = TEMP_DIR / "audio" / filename
    if not path.exists():
        return JSONResponse({"error": "Audio not found"}, status_code=404)
    return FileResponse(str(path), media_type='audio/wav')

# ========== 채팅 API ==========

@router.post("/chat")
async def chat(request: Request):
    """
    채팅 엔드포인트
    - 사용자 메시지를 받아 AI 응답 생성
    - 비디오 생성 작업을 큐에 추가
    """
    data = await request.json()
    text = data.get('text', '')
    session_id = data.get('session_id', str(uuid.uuid4()))
    
    if not text:
        return JSONResponse({"error": "No text provided"}, status_code=400)
    
    # AI 응답 생성
    response = await chat_service.get_response(text, session_id)
    
    # 아바타 생성 작업 큐에 추가
    if response:
        await avatar_generation_service.add_task(AvatarGenerationTask(
            user_text=text,
            ai_response=response,
            session_id=session_id
        ))
    
    return JSONResponse({
        "response": response,
        "session_id": session_id
    })

# ========== WebSocket ==========

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket 연결
    - 비디오 생성 완료 알림용
    - 실시간 상태 업데이트
    """
    await websocket.accept()
    avatar_generation_service.add_connection(websocket)
    
    try:
        while True:
            # 클라이언트 메시지 대기 (연결 유지용)
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass  # 정상 종료
    except Exception:
        pass  # 에러 무시
    finally:
        avatar_generation_service.remove_connection(websocket)
