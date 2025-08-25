"""
Ditto 립싱크 모델 로더
"""
import sys
import torch
from pathlib import Path

# 프로젝트 루트 경로
DITTO_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(DITTO_ROOT))

from stream_pipeline_offline import StreamSDK


class DittoModelPool:
    """Ditto 모델 로더 - SDK 인터페이스 제공"""
    
    def __init__(self):
        """모델 초기화"""
        # TensorRT 버전 사용
        cfg_path = DITTO_ROOT / 'checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl'
        data_root = DITTO_ROOT / 'checkpoints/ditto_trt_Ampere_Plus'
        
        # SDK 초기화
        self.sdk = StreamSDK(str(cfg_path), str(data_root))
        
        # 속도 최적화 설정
        self.SPEED_OVERRIDES = {
            'sampling_timesteps': 5,    # 빠른 샘플링
            'template_n_frames': 1,     # 템플릿 프레임
            'max_size': 512,            # 최대 해상도
        }
    
    def get_sdk(self):
        """SDK 인스턴스 반환"""
        return self.sdk
    
    def get_speed_config(self):
        """속도 설정 반환"""
        return self.SPEED_OVERRIDES
    
    def cleanup(self):
        """GPU 메모리 정리"""
        torch.cuda.empty_cache()
