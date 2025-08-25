#!/usr/bin/env python3
"""
Ditto TalkingHead 실시간 스트리밍 서버
FastAPI + Socket.IO 기반
"""

import os
import sys
import time
import threading
import queue
import base64
import io
import json
import asyncio
# TensorRT 패치
import numpy as np
np.atan2 = np.arctan2
np.int = int
np.float = float
np.bool = bool

from PIL import Image
import cv2
import traceback
from pathlib import Path

# FastAPI
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Socket.IO
import socketio

# 오디오 처리
import librosa
import soundfile as sf

# Ditto SDK 임포트
from stream_pipeline_online import StreamSDK

# FastAPI 앱 생성
app = FastAPI(title="Ditto TalkingHead Streaming Server")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Socket.IO 서버 생성
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
socket_app = socketio.ASGIApp(sio, app)

class DittoStreamingServer:
    def __init__(self):
        """Ditto 실시간 스트리밍 서버 초기화"""
        self.is_streaming = False
        self.sdk = None
        self.frame_queue = asyncio.Queue(maxsize=100)
        self.current_frame_number = 0
        self.total_frames = 0
        self.audio_duration = 0
        
        # 프레임 설정
        self.fps = 25
        self.frame_interval = 1.0 / self.fps
        
        # 현재 프레임
        self.current_frame = None
        self.stream_task = None
        
    def initialize_sdk(self, cfg_pkl='./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl', 
                       data_root='./checkpoints/ditto_trt_Ampere_Plus'):
        """SDK 초기화"""
        try:
            print("🚀 SDK 초기화 중...")
            self.sdk = StreamSDK(cfg_pkl, data_root)
            print("✅ SDK 초기화 완료")
            return True
        except Exception as e:
            print(f"❌ SDK 초기화 실패: {e}")
            traceback.print_exc()
            return False
    
    def setup_stream(self, source_path, **kwargs):
        """스트리밍 설정"""
        if not self.sdk:
            if not self.initialize_sdk():
                raise Exception("SDK 초기화 실패")
        
        try:
            # temp 디렉토리 생성
            os.makedirs("./tmp", exist_ok=True)
            temp_output = "./tmp/temp_streaming_output.mp4"
            
            print(f"📸 소스 이미지: {source_path}")
            self.sdk.setup(source_path, temp_output, **kwargs)
            
            # 커스텀 writer로 교체
            self.sdk.writer = self.CustomFrameCapture(self)
            
            self.is_streaming = True
            self.current_frame_number = 0
            print("✅ 스트리밍 준비 완료")
            
        except Exception as e:
            print(f"❌ 스트리밍 설정 실패: {e}")
            traceback.print_exc()
            raise
    
    def CustomFrameCapture(self, parent):
        """프레임 캡처용 커스텀 writer"""
        class FrameCapture:
            def __init__(self, parent):
                self.parent = parent
                self.frame_count = 0
                
            def __call__(self, img, fmt="rgb"):
                try:
                    if fmt == "bgr":
                        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    else:
                        frame = img
                    
                    # 현재 프레임 업데이트
                    self.parent.current_frame = frame
                    self.parent.current_frame_number += 1
                    self.frame_count += 1
                    
                    # 동기적으로 프레임을 큐에 추가
                    try:
                        self.parent.frame_queue.put_nowait(frame)
                    except:
                        # 큐가 가득 차면 오래된 프레임 제거
                        try:
                            self.parent.frame_queue.get_nowait()
                            self.parent.frame_queue.put_nowait(frame)
                        except:
                            pass
                    
                    # 5프레임마다 로그
                    if self.frame_count % 5 == 0:
                        print(f"📹 프레임 {self.frame_count} 캡처됨")
                        
                except Exception as e:
                    print(f"프레임 캡처 오류: {e}")
                    
            def close(self):
                pass
                
        return FrameCapture(parent)
    
    async def send_frame_to_clients(self, frame):
        """모든 클라이언트에게 프레임 전송"""
        try:
            # PIL Image로 변환
            pil_img = Image.fromarray(frame.astype('uint8'))
            
            # JPEG로 인코딩 (품질 조정으로 크기 최적화)
            buf = io.BytesIO()
            pil_img.save(buf, format='JPEG', quality=75)
            frame_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            # 진행률 계산
            progress = (self.current_frame_number / max(self.total_frames, 1)) * 100
            
            # Socket.IO로 프레임 전송
            await sio.emit('video_frame', {
                'frame': frame_base64,
                'frame_number': self.current_frame_number,
                'progress': min(progress, 100)
            })
            
        except Exception as e:
            print(f"프레임 전송 오류: {e}")
    
    async def process_audio_streaming(self, audio_path):
        """오디오 처리 및 실시간 스트리밍"""
        try:
            print(f"🎵 오디오 파일 로딩: {audio_path}")
            
            # 오디오 로드
            audio_data, sr = librosa.load(audio_path, sr=16000)
            self.audio_duration = len(audio_data) / sr
            self.total_frames = int(self.audio_duration * self.fps)
            
            print(f"⏱️ 오디오 길이: {self.audio_duration:.2f}초")
            print(f"🎬 총 프레임 수: {self.total_frames}")
            
            # 전체 오디오를 Base64로 인코딩해서 클라이언트에 전송
            audio_buffer = io.BytesIO()
            sf.write(audio_buffer, audio_data, sr, format='WAV')
            audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode('utf-8')
            
            # 오디오 데이터 전송
            await sio.emit('audio_data', {
                'audio': audio_base64,
                'sample_rate': sr,
                'duration': self.audio_duration
            })
            print("🔊 오디오 데이터 전송 완료")
            
            # 프레임 전송 태스크 시작
            frame_sender_task = asyncio.create_task(self.frame_sender_loop())
            
            # 1초 단위 청크로 분할
            chunk_size = 16000  # 1초 = 16000 샘플 (16kHz)
            
            for i in range(0, len(audio_data), chunk_size):
                if not self.is_streaming:
                    break
                
                chunk = audio_data[i:i+chunk_size]
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)), 'constant')
                
                print(f"🔄 청크 {i//chunk_size + 1}/{len(audio_data)//chunk_size + 1} 처리 중...")
                
                # SDK에 오디오 청크 전달 (별도 스레드에서)
                await asyncio.get_event_loop().run_in_executor(
                    None, 
                    self.sdk.run_chunk, 
                    chunk, 
                    (3, 5, 2)
                )
                
                # 짧은 대기
                await asyncio.sleep(0.1)
            
            # 프레임 전송 태스크 종료 대기
            frame_sender_task.cancel()
            
            # 스트리밍 완료
            await sio.emit('stream_completed')
            print("✅ 스트리밍 완료")
            
        except Exception as e:
            print(f"❌ 오디오 처리 오류: {e}")
            traceback.print_exc()
            await sio.emit('error', {'message': str(e)})
    
    async def frame_sender_loop(self):
        """프레임을 주기적으로 전송하는 루프"""
        while self.is_streaming:
            try:
                # 큐에서 프레임 가져오기 (비블로킹)
                try:
                    frame = self.frame_queue.get_nowait()
                    await self.send_frame_to_clients(frame)
                except:
                    # 큐가 비어있으면 현재 프레임 사용
                    if self.current_frame is not None:
                        await self.send_frame_to_clients(self.current_frame)
                
                # FPS에 맞춰 대기
                await asyncio.sleep(self.frame_interval)
                
            except Exception as e:
                print(f"프레임 전송 루프 오류: {e}")
                await asyncio.sleep(0.1)
    
    def stop_streaming(self):
        """스트리밍 중지"""
        self.is_streaming = False
        if self.stream_task:
            self.stream_task.cancel()
        if self.sdk:
            try:
                self.sdk.close()
            except:
                pass
        self.current_frame_number = 0
        print("🛑 스트리밍 중지됨")

# 전역 스트리밍 서버 인스턴스
streaming_server = DittoStreamingServer()

# HTML 페이지 라우트
@app.get("/")
async def get_index():
    """메인 HTML 페이지 반환"""
    html_file = Path("index.html")
    if html_file.exists():
        return FileResponse(html_file)
    else:
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Ditto TalkingHead Streaming</title>
        </head>
        <body>
            <h1>index.html 파일을 찾을 수 없습니다.</h1>
            <p>HTML 파일을 같은 디렉토리에 배치해주세요.</p>
        </body>
        </html>
        """)

# Socket.IO 이벤트 핸들러
@sio.event
async def connect(sid, environ):
    """클라이언트 연결"""
    print(f"✅ 클라이언트 연결: {sid}")
    await sio.emit('connected', {'message': '서버 연결 성공'}, room=sid)

@sio.event
async def disconnect(sid):
    """클라이언트 연결 해제"""
    print(f"❌ 클라이언트 연결 해제: {sid}")

@sio.event
async def start_stream(sid, data=None):
    """스트리밍 시작 - 자동으로 base.png와 jenny.wav 사용"""
    try:
        print("▶️ 스트리밍 시작 요청 (자동 파일 선택)")
        
        if streaming_server.is_streaming:
            await sio.emit('error', {'message': '이미 스트리밍 중입니다'}, room=sid)
            return
        
        # 파일 경로 자동 설정
        source_path = './example/base.png'
        audio_path = './example/jenny.wav'
        
        print(f"📸 자동 설정: 소스={source_path}, 오디오={audio_path}")
        
        # 파일 존재 확인
        if not Path(source_path).exists():
            await sio.emit('error', {'message': f'소스 이미지를 찾을 수 없습니다: {source_path}'}, room=sid)
            return
            
        if not Path(audio_path).exists():
            await sio.emit('error', {'message': f'오디오 파일을 찾을 수 없습니다: {audio_path}'}, room=sid)
            return
        
        # 스트리밍 설정
        streaming_server.setup_stream(source_path)
        
        # 스트리밍 시작 알림
        await sio.emit('stream_started')
        
        # 비동기로 오디오 처리 및 스트리밍 시작
        streaming_server.stream_task = asyncio.create_task(
            streaming_server.process_audio_streaming(audio_path)
        )
        
        print("🎬 스트리밍 시작됨")
        
    except Exception as e:
        print(f"❌ 스트리밍 시작 오류: {e}")
        traceback.print_exc()
        await sio.emit('error', {'message': str(e)})

@sio.event
async def stop_stream(sid):
    """스트리밍 중지"""
    try:
        print("⏹️ 스트리밍 중지 요청")
        streaming_server.stop_streaming()
        await sio.emit('stream_stopped')
        print("✅ 스트리밍 중지 완료")
        
    except Exception as e:
        print(f"❌ 스트리밍 중지 오류: {e}")
        await sio.emit('error', {'message': str(e)})

# 서버 실행
if __name__ == "__main__":
    import uvicorn
    
    print("=" * 50)
    print("🎭 Ditto TalkingHead 실시간 스트리밍 서버")
    print("=" * 50)
    print("서버 시작중...")
    print("브라우저에서 http://121.78.147.172:7135 접속")
    print("=" * 50)
    
    # 서버 실행
    uvicorn.run(
        socket_app,  # socketio app 사용
        host="0.0.0.0",
        port=7135,  # 포트 변경
        log_level="info"
    )
