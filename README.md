# Ditto TalkingHead 기반 실시간 AI 아바타 시스템: WebSocket 스트리밍과 트리플 버퍼링을 활용한 립싱크 비디오 생성

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-14-black.svg)](https://nextjs.org/)
[![WebSocket](https://img.shields.io/badge/WebSocket-Streaming-orange.svg)](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket)
[![License](https://img.shields.io/badge/License-Apache%202.0-red.svg)](https://opensource.org/licenses/Apache-2.0)

## 📌 개요

본 저장소는 Ditto TalkingHead 모델을 WebSocket 실시간 스트리밍 및 트리플 버퍼링 기술과 통합하여 구현한 AI 아바타 대화 시스템입니다. 사용자 입력부터 비디오 재생까지 3초 이내의 지연시간으로 끊김 없는 립싱크 비디오 생성을 실현했습니다.

### 🎯 핵심 기술

- **WebSocket 기반 스트리밍**: 비디오 생성 완료 즉시 푸시 알림을 통한 실시간 전송
- **트리플 버퍼링 시스템**: 3개 레이어 순환 재생으로 매끄러운 비디오 전환
- **비동기 처리 파이프라인**: asyncio 기반 비동기 작업 큐로 동시 처리 최적화
- **세션 관리**: Redis/In-memory 기반 대화 히스토리 관리
- **멀티모달 통합**: GPT-4o 대화 생성, Google Cloud TTS 음성 합성, Ditto 립싱크

## 🔥 업데이트

- [2024.12.20] 🔥 트리플 버퍼링 비디오 플레이어 구현 완료
- [2024.12.18] 🔥 WebSocket 자동 재연결 메커니즘 추가
- [2024.12.15] 🔥 비동기 작업 큐 시스템 구현
- [2024.12.10] 🔥 Ditto TalkingHead 모델 통합

## 🏗️ 시스템 아키텍처

### 데이터 처리 흐름

```
사용자 입력 (React)
    ↓
REST API 요청 (Axios)
    ↓
AI 응답 생성 (GPT-4o + LangChain)
    ↓
TTS 음성 합성 (Google Cloud TTS)
    ↓
립싱크 비디오 생성 (Ditto Model Pool)
    ↓
WebSocket 푸시 알림
    ↓
트리플 버퍼링 재생 (React)
```

### 📊 성능 지표

| 메트릭 | 수치 | 설명 |
|--------|------|------|
| **응답 지연시간** | 1-2초 | 사용자 입력 → AI 텍스트 응답 |
| **비디오 생성** | 2-3초 | 음성 합성 → 립싱크 비디오 |
| **WebSocket 지연** | <50ms | 서버 → 클라이언트 푸시 |
| **동시 세션** | 10+ | 멀티 세션 동시 처리 가능 |
| **비디오 전환** | 300ms | 페이드 애니메이션 시간 |
| **메모리 사용** | 6GB | GPU VRAM 사용량 |

## 🛠️ 설치 및 실행

### 시스템 요구사항

- **System**: Ubuntu 20.04 / Windows 10+
- **GPU**: NVIDIA RTX 3060+ (6GB+ VRAM)
- **Python**: 3.10+
- **Node.js**: 18+
- **CUDA**: 12.1
- **FFmpeg**: 4.4+

### 환경 설정

저장소 클론:
```bash
git clone https://github.com/yourusername/ditto-ai-avatar.git
cd ditto-ai-avatar
```

### Backend 설정

#### Conda 환경
```bash
conda env create -f environment.yaml
conda activate ditto
```

#### Poetry 환경 (TensorRT 지원)
```bash
poetry install
poetry shell
```

#### 환경 변수 설정
```bash
cp .env.example .env
# .env 파일에 아래 키 설정:
# OPENAI_API_KEY=your_openai_api_key
# TTS_API_KEY=your_google_cloud_tts_key
# USE_REDIS=false  # Redis 사용 여부
```

#### 체크포인트 다운로드
```bash
# HuggingFace에서 Ditto 모델 다운로드
git lfs install
git clone https://huggingface.co/digital-avatar/ditto-talkinghead checkpoints
```

체크포인트 구조:
```
./checkpoints/
├── ditto_cfg/
│   ├── v0.4_hubert_cfg_trt.pkl      # 오프라인 설정
│   └── v0.4_hubert_cfg_trt_online.pkl  # 온라인 설정
├── ditto_onnx/                      # ONNX 모델
└── ditto_trt_Ampere_Plus/           # TensorRT 엔진
```

#### 서버 실행
```bash
python src/main.py  # http://localhost:7136
```

### Frontend 설정

```bash
cd frontend
npm install
npm run dev  # http://localhost:7134
```

### Docker 실행

```bash
docker-compose up -d
```

## 🚀 사용 방법

1. 브라우저에서 `http://localhost:7134` 접속
2. 자동으로 새 세션 생성 및 리다이렉트
3. 채팅 입력창에 메시지 입력
4. AI 응답이 텍스트로 즉시 표시
5. 비디오가 생성되면 자동으로 재생

## 💻 핵심 구현

### WebSocket 실시간 스트리밍

```python
# services.py - 비동기 비디오 처리 워커
async def process_queue_worker(self):
    while True:
        task = await self.video_queue.get()
        
        # 백그라운드에서 TTS 및 비디오 생성
        result = await asyncio.to_thread(
            self.processor.process_message_with_audio,
            task.user_text, task.ai_response
        )
        
        # WebSocket으로 모든 클라이언트에 브로드캐스트
        await self.broadcast({
            'type': 'video_ready',
            'data': result
