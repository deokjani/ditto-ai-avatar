# Ditto TalkingHead 기반 실시간 AI 아바타 시스템: TensorRT 최적화와 WebSocket 스트리밍

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![TensorRT](https://img.shields.io/badge/TensorRT-8.6.1-76B900.svg)](https://developer.nvidia.com/tensorrt)
[![WebSocket](https://img.shields.io/badge/WebSocket-Streaming-orange.svg)](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket)
[![License](https://img.shields.io/badge/License-Apache%202.0-red.svg)](https://opensource.org/licenses/Apache-2.0)

<br>

https://github.com/user-attachments/assets/demo-video.mp4

✨ **실시간 립싱크 비디오 생성 데모** ✨

</div>

## **📌 개요**

Kling AI로 생성한 아바타 비디오를 기반으로 Google Cloud TTS와 Ditto TalkingHead 모델을 통합하여 실시간 립싱크 비디오를 생성하는 시스템입니다. TensorRT 최적화와 파라미터 튜닝으로 실시간 처리 성능을 달성했으며, WebSocket을 통한 비디오 완료 알림으로 끊김 없는 재생을 구현했습니다.

### **🎯 핵심 성과**

- **TensorRT 최적화**: PyTorch 대비 4.5배 빠른 추론 속도 달성
- **파라미터 튜닝**: sampling_timesteps 10, max_size 512, template_n_frames 3로 속도 최적화
- **WebSocket 알림**: 비디오 생성 완료 즉시 푸시 알림
- **트리플 버퍼링**: 3개 비디오 레이어 순환으로 끊김 없는 전환
- **멀티모달 통합**: GPT-4o + Google TTS + Ditto TalkingHead 파이프라인

## **🛠️ Installation**

### **Tested Environment**

* **System**: Ubuntu 22.04.4 LTS (Jammy Jellyfish)
* **GPU**: NVIDIA A100 80GB PCIe
* **Python**: 3.10
* **CUDA**: 12.8
* **TensorRT**: 8.6.1

## **📊 성능 벤치마크**

### **PyTorch vs TensorRT 실측 비교**

```
📈 DIT (Diffusion) Model 처리 속도
   PyTorch:  ████████░░░░░░░░░░░░  3.4 it/s (평균)
   TensorRT: ████████████████████  15.2 it/s (평균)
   속도 향상: 4.5배 ⬆

📹 Video Writer 처리 속도  
   PyTorch:  ████████████░░░░░░░░  20 it/s (평균)
   TensorRT: ████████████████████  33 it/s (평균)
   속도 향상: 1.65배 ⬆

⏱️ 실제 처리 시간
   전체 비디오 생성: PyTorch 2-3초 → TensorRT 1-1.5초
   실시간 FPS: PyTorch 3 FPS → TensorRT 15 FPS
```

### **실측 로그**
```
# PyTorch
dit: 1it [00:00, 3.17it/s]
dit: 1it [00:00, 3.66it/s] (두 번째 실행)
writer: 25it [00:01, 17.37it/s]
writer: 59it [00:02, 23.95it/s]

# TensorRT  
dit: 1it [00:00, 15.25it/s]
dit: 1it [00:00, 15.10it/s] (두 번째 실행)
writer: 25it [00:00, 30.34it/s]
writer: 58it [00:01, 35.40it/s]
```

## **⚡ 성능 최적화**

### **모델 설정 비교**

```python
# model_pool.py

# PyTorch 버전 (기본)
# cfg_path = 'checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl'
# data_root = 'checkpoints/ditto_pytorch'

# TensorRT 버전 (최적화) - 4.5배 빠름
cfg_path = 'checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl'
data_root = 'checkpoints/ditto_trt_Ampere_Plus'
```

### **속도 최적화 파라미터**

```python
# model_pool.py - 파라미터 튜닝
SPEED_OVERRIDES = {
    'sampling_timesteps': 10,    # 50 → 10 (5배 단축)
    'template_n_frames': 3,       # 최소 프레임으로 초기화 시간 단축
    'max_size': 512,             # 1920 → 512 (해상도 조정으로 속도 향상)
}
```

## **🏗️ 시스템 아키텍처**

```
Kling AI 아바타 비디오 (base.mp4)
         ↓
사용자 입력 (HTML/JavaScript)
         ↓
REST API 요청 → AI 응답 생성 (GPT-4o)
         ↓
asyncio.Queue (비동기 작업 큐)
         ↓
백그라운드 워커:
  ├─ Google Cloud TTS (음성 합성)
  └─ Ditto TalkingHead (립싱크 생성)
         ↓
WebSocket 비디오 완료 알림
         ↓
트리플 버퍼링 비디오 재생 (JavaScript)
```

## **💻 핵심 구현**

### **1. TensorRT 모델 초기화 및 최적화**

```python
# model_pool.py - TensorRT 최적화 설정
class DittoModelPool:
    def __init__(self):
        # TensorRT 엔진 사용
        cfg_path = DITTO_ROOT / 'checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl'
        data_root = DITTO_ROOT / 'checkpoints/ditto_trt_Ampere_Plus'
        
        # SDK 초기화
        self.sdk = StreamSDK(str(cfg_path), str(data_root))
        
        # 속도 최적화 파라미터 적용
        self.SPEED_OVERRIDES = {
            'sampling_timesteps': 10,
            'template_n_frames': 3,
            'max_size': 512,
        }
        
    def process(self, audio_path, video_path, output_path):
        # Kling AI 비디오 + TTS 오디오 → 립싱크 비디오
        return self.sdk.run(
            source_path=video_path,  # Kling AI 아바타
            audio_path=audio_path,   # Google TTS 음성
            output_path=output_path,
            **self.SPEED_OVERRIDES
        )
```

### **2. Google Cloud TTS 통합**

```python
# video_processor.py - TTS 생성
def google_tts(self, text: str, output_path: str):
    response = requests.post(
        "https://texttospeech.googleapis.com/v1/text:synthesize",
        headers={"X-Goog-Api-Key": TTS_API_KEY},
        json={
            "input": {"text": text},
            "voice": {
                "languageCode": "en-US",
                "name": "en-US-Chirp-HD-F",
            },
            "audioConfig": {
                "audioEncoding": "LINEAR16",
                "sampleRateHertz": 16000,
                "speakingRate": 1.1,  # 약간 빠르게 (1.0 → 1.1)
            }
        }
    )
    
    # 립싱크 동기화를 위한 패딩
    silence_front = np.zeros(int(16000 * 0.15), dtype=np.float32)  # 앞 0.15초
    silence_back = np.zeros(int(16000 * 0.1), dtype=np.float32)    # 뒤 0.1초
    padded_audio = np.concatenate([silence_front, audio_array, silence_back])
```

### **3. WebSocket 비디오 완료 알림**

```python
# services.py - 비디오 생성 완료 시 WebSocket 푸시
async def process_queue_worker(self):
    while True:
        task = await self.video_queue.get()
        
        # TTS + Ditto 처리 (별도 스레드)
        result = await asyncio.to_thread(
            self.processor.process_message_with_audio,
            task.user_text,
            task.ai_response
        )
        
        # WebSocket으로 비디오 완료 알림
        await self.broadcast({
            'type': 'video_ready',
            'data': {
                'video_url': f"/video/{Path(result['video_path']).name}",
                'audio_url': f"/audio/{Path(result['audio_path']).name}",
                'duration': result['duration']
            }
        })
```

### **4. 트리플 버퍼링 비디오 플레이어 (JavaScript)**

```javascript
// main.js - 3개 레이어 순환 재생
class VideoManager {
    constructor() {
        this.layers = [
            { element: document.getElementById('videoLayer1'), video: document.getElementById('video1') },
            { element: document.getElementById('videoLayer2'), video: document.getElementById('video2') },
            { element: document.getElementById('videoLayer3'), video: document.getElementById('video3') }
        ];
        this.currentLayerIndex = 0;
    }
    
    async switchToVideo(url, videoType, audioUrl) {
        const nextLayerIndex = (this.currentLayerIndex + 1) % 3;
        const nextLayer = this.layers[nextLayerIndex];
        
        // 다음 레이어에 비디오 로드
        nextLayer.video.src = url;
        nextLayer.video.loop = (videoType === 'default');
        await nextLayer.video.play();
        
        // TTS 오디오 동기화
        if (audioUrl && videoType === 'response') {
            this.audioPlayer.src = audioUrl;
            this.audioPlayer.play();
        }
        
        // CSS 페이드 전환
        nextLayer.element.classList.add('active');
        
        // 300ms 후 이전 레이어 제거
        setTimeout(() => {
            this.layers[this.currentLayerIndex].element.classList.remove('active');
            this.currentLayerIndex = nextLayerIndex;
        }, 300);
    }
}
```

### **5. WebSocket 연결 관리 (JavaScript)**

```javascript
// main.js - WebSocket 비디오 알림 수신
function connectWebSocket() {
    const wsUrl = `ws://${window.location.host}/ws`;
    socket = new WebSocket(wsUrl);
    
    socket.onopen = function() {
        // 30초마다 ping으로 연결 유지
        setInterval(() => {
            if (socket.readyState === WebSocket.OPEN) {
                socket.send('ping');
            }
        }, 30000);
    };
    
    socket.onmessage = function(event) {
        const data = JSON.parse(event.data);
        if (data.type === 'video_ready') {
            // 비디오 큐에 추가
            videoManager.addToQueue({
                url: data.data.video_url,
                audioUrl: data.data.audio_url
            });
        }
    };
    
    socket.onclose = function() {
        setTimeout(connectWebSocket, 3000);  // 3초 후 재연결
    };
}
```

## **🚀 빠른 시작**

### **1. 저장소 클론**
```bash
git clone https://github.com/yourusername/ditto-ai-avatar.git
cd ditto-ai-avatar
```

### **2. 체크포인트 다운로드**
```bash
# HuggingFace에서 Ditto 모델 다운로드
git lfs install
git clone https://huggingface.co/digital-avatar/ditto-talkinghead checkpoints
```

### **3. 환경 설정**

#### **TensorRT 버전 (권장)**
```bash
# Poetry로 TensorRT 환경 설정
poetry install
poetry shell

# 환경 변수 설정
cp .env.example .env
# OPENAI_API_KEY, TTS_API_KEY 설정
```

#### **PyTorch 버전**
```bash
# Conda로 PyTorch 환경 설정
conda env create -f environment.yaml
conda activate ditto
```

### **4. 서버 실행**
```bash
python src/main.py  # http://localhost:7136
```

브라우저에서 `http://localhost:7136` 접속

## **📁 프로젝트 구조**

```
ditto-ai-avatar/
├── src/
│   ├── main.py              # FastAPI 앱
│   ├── api.py               # REST/WebSocket 엔드포인트
│   ├── services.py          # 비동기 서비스 (Queue, WebSocket)
│   ├── video_processor.py   # Google TTS + 비디오 생성
│   ├── model_pool.py        # Ditto TensorRT/PyTorch 관리
│   └── llm_chain.py         # GPT-4o 체인
│
├── templates/
│   └── index.html           # 메인 HTML 페이지
│
├── static/
│   ├── js/
│   │   └── main.js          # 트리플 버퍼링 비디오 플레이어
│   └── css/
│       └── style.css        # 스타일시트
│
├── checkpoints/
│   ├── ditto_cfg/           # 설정 파일
│   ├── ditto_pytorch/       # PyTorch 모델
│   └── ditto_trt_Ampere_Plus/  # TensorRT 엔진
│
├── example/
│   └── base.mp4            # Kling AI 아바타 비디오
│
└── docker-compose.yml
```

## **🔧 기술적 특징**

### **TensorRT 최적화**
- FP16 정밀도로 메모리 50% 절감
- 커널 퓨전으로 연산 최적화
- A100 GPU 최적화 (Ampere_Plus)

### **비동기 처리**
- FastAPI + asyncio 완전 비동기 구조
- asyncio.Queue 기반 백그라운드 작업
- asyncio.to_thread로 CPU/GPU 작업 분리

### **WebSocket 통신**
- 비디오 생성 완료 알림 전용
- 자동 재연결 메커니즘 (3초)
- 30초 ping/pong 연결 유지

### **비디오 재생**
- 트리플 버퍼링으로 끊김 제거
- CSS transition 페이드 효과 (300ms)
- 비디오 큐 순차 재생

## **📊 성능 지표**

| 메트릭 | 수치 | 설명 |
|--------|------|------|
| **텍스트 응답** | 1-2초 | GPT-4o 응답 시간 |
| **TTS 생성** | 0.5초 | Google Cloud TTS |
| **립싱크 생성** | 1-1.5초 | Ditto TensorRT |
| **전체 파이프라인** | 2-3초 | 입력→비디오 재생 |
| **GPU 메모리** | 12GB | A100 VRAM 사용량 |
| **동시 세션** | 10+ | 멀티 세션 지원 |

## **🐛 문제 해결**

### NumPy 호환성
```python
# main.py - NumPy 1.26.4 패치
import numpy as np
np.atan2 = np.arctan2
```

### TensorRT 변환
```bash
# GPU가 Ampere_Plus를 지원하지 않는 경우
python scripts/cvt_onnx_to_trt.py \
    --onnx_dir "./checkpoints/ditto_onnx" \
    --trt_dir "./checkpoints/ditto_trt_custom"
```

## **📝 TODO**

- [ ] 한국어 TTS 지원
- [ ] 감정 표현 파라미터 추가
- [ ] WebRTC P2P 스트리밍
- [ ] 모바일 반응형 UI
- [ ] Kubernetes 배포

## **📄 라이선스**

Apache License 2.0

## **🙏 감사의 글**

- **Ant Group**: Ditto TalkingHead 모델 제공
- **Kling AI**: 고품질 아바타 비디오 생성
- **Google Cloud**: TTS API
- **OpenAI**: GPT-4o 모델

---

<div align="center">

**⭐ Star를 눌러주시면 큰 힘이 됩니다!**

Built with TensorRT, WebSocket, and Ditto TalkingHead

</div>
