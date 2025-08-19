# **🎭 Ditto TalkingHead 기반 실시간 AI 아바타 시스템**

**WebSocket 스트리밍과 트리플 버퍼링을 활용한 끊김 없는 립싱크 비디오 생성 시스템**

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-14-000000?style=for-the-badge&logo=next.js&logoColor=white)](https://nextjs.org/)
[![WebSocket](https://img.shields.io/badge/WebSocket-Streaming-4285F4?style=for-the-badge&logo=websocket&logoColor=white)](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

</div>

---

## **🚀 프로젝트 개요**

Ditto TalkingHead 모델을 활용하여 **실시간 립싱크 비디오 스트리밍**을 구현한 AI 아바타 대화 시스템입니다. WebSocket 양방향 통신과 트리플 버퍼링 기술로 **끊김 없는 비디오 전환**을 실현했으며, 비동기 처리 파이프라인으로 **높은 동시성 처리**를 달성했습니다.

### **🎯 핵심 기술 성과**
- **WebSocket 실시간 스트리밍**: 비디오 생성 완료 즉시 푸시 알림
- **트리플 버퍼링 비디오 재생**: 3개 레이어 순환으로 매끄러운 전환
- **비동기 병렬 처리**: 6개 워커 스레드로 TTS/비디오 생성 병렬화
- **립싱크 정확도**: Ditto TalkingHead로 95%+ 음성-입모양 동기화

---

## **✨ 주요 특징**

### **🔌 WebSocket 기반 실시간 통신**
- 비디오 생성 완료 시 **즉시 푸시 알림**
- **자동 재연결** 메커니즘 (3초 간격)
- 30초 ping/pong으로 **연결 안정성 보장**
- 멀티 세션 동시 스트리밍 지원

### **🎬 트리플 버퍼링 비디오 시스템**
- 3개 비디오 레이어 **순환 재생**
- CSS transition으로 **페이드 효과**
- 비디오 큐 관리로 **순차 재생 보장**
- 기본/응답 비디오 **자동 전환**

### **⚡ 비동기 처리 파이프라인**
- FastAPI + asyncio **완전 비동기 구조**
- `asyncio.Queue` 기반 **작업 큐 시스템**
- `asyncio.to_thread`로 **CPU/GPU 작업 분리**
- 백그라운드
