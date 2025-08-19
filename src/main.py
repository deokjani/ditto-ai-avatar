"""
메인 진입점
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# TensorRT 패치
import numpy as np
np.atan2 = np.arctan2
np.int = int
np.float = float
np.bool = bool

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from services import video_service
import api

# 경로 설정 
PROJECT_ROOT = Path(__file__).parent.parent
STATIC_DIR = PROJECT_ROOT / "static"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작
    await video_service.initialize()
    worker = asyncio.create_task(video_service.process_queue_worker())
    yield
    # 종료
    worker.cancel()
    try:
        await worker
    except asyncio.CancelledError:
        pass

# FastAPI 앱
app = FastAPI(title="Ditto TalkingHead", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# 정적 파일
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# 라우터
app.get("/")(api.index)
app.get("/health")(api.health_check)
app.get("/default_video")(api.get_default_video)
app.get("/video/{filename}")(api.get_video)
app.get("/audio/{filename}")(api.get_audio)
app.post("/chat")(api.chat)  # 채팅 REST API 추가
app.websocket("/ws")(api.websocket_endpoint)  # 비디오 스트리밍용

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7136, log_level="info")
