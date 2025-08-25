"""
Ditto TalkingHead API
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
from v1.router import router
from v1.services import avatar_generation_service

# 경로 설정
STATIC_DIR = Path(__file__).parent.parent / "static"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """생명주기 관리"""
    await avatar_generation_service.initialize()
    worker = asyncio.create_task(avatar_generation_service.process_queue_worker())
    yield
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

# 라우터 등록
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7136, log_level="info")
