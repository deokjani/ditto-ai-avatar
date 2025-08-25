"""
서비스 계층 - 비즈니스 로직
"""
import asyncio
from typing import List
from pathlib import Path
from fastapi import WebSocket

from v1.schemas import AvatarGenerationTask, AvatarGenerationResult
from v1.avatar_generation.video_pipeline import VideoPipeline
from v1.chat.llm_chain import talking_chain, clear_memory_sessions
import v1.config as config


class ChatService:
    """채팅 서비스 - AI 대화 처리"""
    
    def __init__(self):
        self.chain = talking_chain()
        self.message_count = 0
    
    async def get_response(self, user_text: str, session_id: str):
        """
        AI 응답 생성
        
        Args:
            user_text: 사용자 입력
            session_id: 세션 ID
            
        Returns:
            str: AI 응답 텍스트
        """
        # 인메모리 사용 시 주기적 정리
        if not config.USE_REDIS:
            self.message_count += 1
            if self.message_count % 20 == 0:
                clear_memory_sessions()
        
        # LLM 호출
        response = await self.chain.ainvoke(
            {"query": user_text},
            config={"configurable": {"session_id": session_id}}
        )
        return response


class AvatarGenerationService:
    """아바타 생성 서비스 - TTS 및 립싱크 비디오 생성"""
    
    def __init__(self):
        self.processor = None
        self.video_queue = asyncio.Queue()
        self.connections: List[WebSocket] = []
    
    async def initialize(self):
        """서비스 초기화"""
        print("Initializing avatar generation service...")
        self.processor = await asyncio.to_thread(VideoPipeline)
        print("Avatar generation service initialized")
        
        # 설정 정보 출력
        print("="*50)
        if config.USE_REDIS:
            print(f"Session Storage: Redis ({config.REDIS_URL})")
        else:
            print("Session Storage: In-Memory (max 50 sessions)")
        print("="*50)
    
    def add_connection(self, ws: WebSocket):
        """WebSocket 연결 추가"""
        if ws not in self.connections:
            self.connections.append(ws)
    
    def remove_connection(self, ws: WebSocket):
        """WebSocket 연결 제거"""
        if ws in self.connections:
            self.connections.remove(ws)
    
    async def broadcast(self, message: dict):
        """모든 클라이언트에 메시지 전송"""
        for conn in self.connections[:]:
            try:
                await conn.send_json(message)
            except:
                self.remove_connection(conn)
    
    async def add_task(self, task: AvatarGenerationTask):
        """생성 작업을 큐에 추가"""
        await self.video_queue.put(task)
    
    async def process_queue_worker(self):
        """백그라운드 워커 - 큐 처리"""
        # 프로세서 초기화 대기
        while not self.processor:
            await asyncio.sleep(0.5)
        
        print("Avatar generation worker started")
        
        # 큐 처리 루프
        while True:
            try:
                task = await self.video_queue.get()
                
                # 아바타 비디오 생성
                result = await asyncio.to_thread(
                    self.processor.process_message_with_audio,
                    task.user_text,
                    task.ai_response
                )
                
                # 결과 포맷팅
                result_data = AvatarGenerationResult(
                    video_url=f"/video/{Path(result['video_path']).name}",
                    audio_url=f"/audio/{Path(result['audio_path']).name}" if result.get('audio_path') else None,
                    duration=result['duration'],
                    text=task.ai_response
                )
                
                # 클라이언트에 알림
                await self.broadcast({
                    'type': 'video_ready',
                    'data': result_data.dict()
                })
                
            except Exception as e:
                print(f"Error in avatar generation: {e}")
                await asyncio.sleep(1)


# 전역 인스턴스
chat_service = ChatService()
avatar_generation_service = AvatarGenerationService()
