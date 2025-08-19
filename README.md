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

<table>
<tr>
<th>메트릭</th>
<th>PyTorch</th>
<th>TensorRT</th>
<th>성능 향상</th>
</tr>
<tr>
<td><b>DIT Model</b></td>
<td>3.4 it/s</td>
<td>15.2 it/s</td>
<td><b>🚀 4.5배</b></td>
</tr>
<tr>
<td><b>Video Writer</b></td>
<td>20 it/s</td>
<td>33 it/s</td>
<td><b>⚡ 1.65배</b></td>
</tr>
<tr>
<td><b>전체 생성 시간</b></td>
<td>2-3초</td>
<td>1-1.5초</td>
<td><b>⏱️ 2배</b></td>
</tr>
<tr>
<td><b>실시간 FPS</b></td>
<td>3 FPS</td>
<td>15 FPS</td>
<td><b>🎯 5배</b></td>
</tr>
</table>

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
REST API 요청 (/chat)
         ↓
AI 응답 생성 (GPT-4o + LangChain)
         ↓
즉시 텍스트 응답 반환 (사용자에게 표시)
         ↓
asyncio.Queue에 비디오 작업 추가
         ↓
백그라운드 워커에서 처리:
  ├─ Google Cloud TTS (음성 합성)
  └─ Ditto TalkingHead (립싱크 생성)
         ↓
WebSocket으로 비디오 완료 알림
         ↓
트리플 버퍼링 비디오 재생 (JavaScript)
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
python src/main.py  # http://localhost:7135
```

브라우저에서 `http://localhost:7135` 접속

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
    └── base.mp4            # Kling AI 아바타 비디오
```

## **🔧 기술적 특징**

### **TensorRT 최적화**
- **FP16 정밀도**: 32비트 → 16비트로 메모리 50% 절감
- **커널 퓨전**: 여러 연산을 하나로 합쳐 메모리 접근 최소화
- **Ampere_Plus**: A100 GPU 아키텍처에 최적화된 엔진

### **비동기 처리**
- FastAPI + asyncio 완전 비동기 구조
- asyncio.Queue 기반 백그라운드 작업
- asyncio.to_thread로 CPU/GPU 작업 분리

### **WebSocket 통신**
- 비디오 생성 완료 알림 전용
- 자동 재연결 메커니즘 (3초)
- 30초 ping/pong으로 연결 유지 (타임아웃 방지)

### **비디오 재생**
- 트리플 버퍼링으로 끊김 제거
- CSS transition 페이드 효과 (300ms)
- z-index 레이어링으로 순차 재생
- 비디오 큐 자동 관리

## **📊 성능 지표**

| 메트릭 | 수치 | 설명 |
|--------|------|------|
| **텍스트 응답** | 1-2초 | GPT-4o 응답 시간 |
| **TTS 생성** | 0.5초 | Google Cloud TTS |
| **립싱크 생성** | 1-1.5초 | Ditto TensorRT |
| **전체 파이프라인** | 2-3초 | 입력→비디오 재생 |
| **GPU 메모리** | 12GB | A100 VRAM 사용량 |

### **PyTorch vs TensorRT 모델 전환시 문제점**

#### **NumPy 호환성 해결**
TensorRT 사용 시 NumPy 2.0과의 호환성 문제 해결:
```python
# main.py - TensorRT와 NumPy 1.26.4 호환성 패치
import numpy as np
np.atan2 = np.arctan2  # TensorRT가 사용하는 deprecated 함수 매핑
np.int = int           # NumPy 2.0에서 제거된 타입 복원
np.float = float
np.bool = bool
```


## **📄 라이선스**

Apache License 2.0

## **🙏 감사의 글**

- **Ant Group**: Ditto TalkingHead 모델 제공
- **Kling AI**: 고품질 아바타 비디오 생성
- **Google Cloud**: TTS API
- **OpenAI**: GPT-4o 모델

---

