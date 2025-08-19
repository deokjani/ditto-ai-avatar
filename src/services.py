"""
서비스 클래스들
"""
import asyncio
from typing import List
from pathlib import Path
from fastapi import WebSocket

from schemas import VideoTask, VideoResult
from video_processor import VideoProcessor
from llm_chain import talking_chain, clear_memory_sessions
import params

class ChatService:
    """채팅 서비스"""
    def __init__(self):
        self.chain = talking_chain()
        self.message_count = 0
    
    async def get_response(self, user_text: str, session_id: str):
        """AI 응답 반환 (invoke 방식)"""
        # 인메모리 사용 시 20번 요청마다 정리
        if not params.USE_REDIS:
            self.message_count += 1
            if self.message_count % 20 == 0:
                clear_memory_sessions()
        
        # invoke로 전체 응답을 한 번에 받음
        response = await self.chain.ainvoke(
            {"query": user_text},
            config={"configurable": {"session_id": session_id}}
        )
        return response

class VideoService:
    """비디오 생성 서비스"""
    def __init__(self):
        self.processor = None
        self.video_queue = asyncio.Queue()
        self.connections: List[WebSocket] = []
    
    async def initialize(self):
        """비디오 프로세서 초기화"""
        print("Initializing video processor...")
        self.processor = await asyncio.to_thread(VideoProcessor)
        print("Video processor initialized")
        
        # 세션 저장소 모드 출력
        print("="*50)
        if params.USE_REDIS:
            print(f"Session Storage: Redis ({params.REDIS_URL})")
        else:
            print("Session Storage: In-Memory (max 50 sessions)")
        print("="*50)
    
    def add_connection(self, ws: WebSocket):
        if ws not in self.connections:
            self.connections.append(ws)
    
    def remove_connection(self, ws: WebSocket):
        if ws in self.connections:
            self.connections.remove(ws)
    
    async def broadcast(self, message: dict):
        """모든 연결에 메시지 전송"""
        for conn in self.connections[:]:
            try:
                await conn.send_json(message)
            except:
                self.remove_connection(conn)
    
    async def add_task(self, task: VideoTask):
        """비디오 작업 추가"""
        await self.video_queue.put(task)
    
    async def process_queue_worker(self):
        """백그라운드 워커"""
        while not self.processor:
            await asyncio.sleep(0.5)
        
        print("Video queue worker started")
        
        while True:
            try:
                task = await self.video_queue.get()
                
                # 비디오 생성
                result = await asyncio.to_thread(
                    self.processor.process_message_with_audio,
                    task.user_text,
                    task.ai_response
                )
                
                # 결과 전송
                video_result = VideoResult(
                    video_url=f"/video/{Path(result['video_path']).name}",
                    audio_url=f"/audio/{Path(result['audio_path']).name}" if result.get('audio_path') else None,
                    duration=result['duration'],
                    text=task.ai_response
                )
                
                await self.broadcast({
                    'type': 'video_ready',
                    'data': video_result.dict()
                })
            except Exception as e:
                print(f"Error in video processing: {e}")
                await asyncio.sleep(1)

# 전역 인스턴스
chat_service = ChatService()
video_service = VideoService()
