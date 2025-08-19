"""
model_pool.py - AI 모델 풀 관리
Ditto 립싱크 모델을 관리하고 오디오-비디오 동기화 처리
"""
import sys
import time
import tempfile
import numpy as np
import torch
from pathlib import Path

# 프로젝트 루트 경로 설정
DITTO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(DITTO_ROOT))

from stream_pipeline_offline import StreamSDK  # Ditto 스트리밍 SDK


class DittoModelPool:
    """
    Ditto 모델 풀 클래스
    립싱크 AI 모델을 로드하고 오디오와 비디오를 동기화
    """
    
    def __init__(self):
        """
        모델 초기화
        체크포인트와 설정 파일 로드
        """
        # 모델 설정 파일 경로
        # cfg_path = DITTO_ROOT / 'checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl'
        # 모델 가중치 파일 경로
        # data_root = DITTO_ROOT / 'checkpoints/ditto_pytorch'
        
        # TensorRT 버전 사용
        cfg_path = DITTO_ROOT / 'checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl'
        data_root = DITTO_ROOT / 'checkpoints/ditto_trt_Ampere_Plus'
    
        # Ditto SDK 초기화
        self.sdk = StreamSDK(str(cfg_path), str(data_root))
        
        # 속도 최적화 설정 (최소한의 필수 파라미터만)
        self.SPEED_OVERRIDES = {
            # 속도 설정
            'sampling_timesteps': 10,       # 빠른 샘플링
            'template_n_frames': 3,        # 프레임 수
            'max_size': 512,               # 해상도 조정
        }
    
    def process(self, audio_path, source_path, output_path):
        """
        오디오와 소스 비디오를 사용해 립싱크 비디오 생성
        
        Args:
            audio_path: 입력 오디오 파일 경로 (TTS 음성)
            source_path: 소스 비디오/이미지 경로 (아바타)
            output_path: 출력 비디오 파일 경로
            
        Returns:
            str: 생성된 비디오 파일 경로 (성공 시) 또는 None
        """
        import librosa
        import soundfile as sf
        import os
        
        # 오디오 로드 (16kHz 샘플링 레이트로 변환)
        audio, _ = librosa.load(audio_path, sr=16000, dtype=np.float32)
        
        # SDK 설정 (소스와 출력 경로 지정)
        self.sdk.setup(source_path, output_path, **self.SPEED_OVERRIDES)
        
        # 오디오 길이에 따른 프레임 수 계산
        audio_duration = len(audio) / 16000  # 오디오 길이 (초)
        num_frames = min(int(audio_duration * 25) + 10, 300) 
        
        # 프레임 수 설정 (페이드 효과 없음)
        self.sdk.setup_Nd(
            N_d=num_frames,    # 생성할 프레임 수
            fade_in=-1,        # 페이드인 없음
            fade_out=-1,       # 페이드아웃 없음
            ctrl_info={}       # 추가 제어 정보 없음
        )
        
        # 오디오를 특징 벡터로 변환 (HuBERT 모델 사용)
        aud_feat = self.sdk.wav2feat.wav2feat(audio, sr=16000)
        
        # 오디오 특징을 모션 생성 큐에 추가
        # SDK가 비동기적으로 립싱크 비디오 생성
        self.sdk.audio2motion_queue.put(aud_feat)
        
        # SDK 종료 (생성 완료 대기)
        self.sdk.close()
        
        # FFmpeg로 오디오와 비디오 병합
        # SDK는 비디오만 생성하므로 원본 오디오를 다시 합성
        tmp_output = self.sdk.tmp_output_path  # 임시 비디오 파일
        cmd = (
            f'ffmpeg -loglevel error -y '              # 조용히, 덮어쓰기
            f'-i "{tmp_output}" '                      # 입력: 생성된 비디오
            f'-i "{audio_path}" '                      # 입력: 원본 오디오
            f'-map 0:v -map 1:a '                      # 비디오는 첫 번째, 오디오는 두 번째
            f'-c:v copy -c:a aac '                     # 비디오 복사, 오디오 AAC 인코딩
            f'-af "aresample=async=1:first_pts=0" '    # 오디오 동기화
            f'"{output_path}"'                         # 최종 출력
        )
        os.system(cmd)
        
        # 생성된 파일 경로 반환 (실패 시 None)
        return str(output_path) if Path(output_path).exists() else None
    
    def cleanup(self):
        """
        GPU 메모리 정리
        모델 사용 후 CUDA 캐시 비우기
        """
        torch.cuda.empty_cache()
