"""
아바타 비디오 생성 파이프라인
"""
import os
import time
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path

from v1.avatar_generation.tts_processor import TTSProcessor
from v1.avatar_generation.lipsync_model import DittoModelPool


class VideoPipeline:
    """비디오 생성 파이프라인 - TTS → 립싱크 → 최종 비디오"""
    
    def __init__(self):
        """초기화 및 모델 로드"""
        # 경로 설정
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.base_video = self.project_root / "example" / "base.mp4"
        self.temp_dir = Path.home() / ".cache" / "ditto_temp"
        
        # 디렉토리 생성
        (self.temp_dir / "videos").mkdir(parents=True, exist_ok=True)
        (self.temp_dir / "audio").mkdir(parents=True, exist_ok=True)
        (self.temp_dir / "tts").mkdir(parents=True, exist_ok=True)
        
        # 컴포넌트 초기화
        self.tts_processor = TTSProcessor()
        self.lipsync_model = DittoModelPool()
        self.sdk = self.lipsync_model.get_sdk()
        self.speed_config = self.lipsync_model.get_speed_config()
        
        # 워밍업
        self._warmup()
    
    def _warmup(self):
        """모델 워밍업 - 첫 실행 속도 개선"""
        try:
            warmup_audio = np.zeros(16000, dtype=np.float32)
            warmup_path = self.temp_dir / "warmup.wav"
            sf.write(str(warmup_path), warmup_audio, 16000)
            
            warmup_output = self.temp_dir / "warmup.mp4"
            self._generate_lipsync_video(
                str(warmup_path),
                str(self.base_video),
                str(warmup_output)
            )
            
            warmup_path.unlink(missing_ok=True)
            warmup_output.unlink(missing_ok=True)
        except:
            pass
    
    def process_message_with_audio(self, user_text: str, ai_response: str):
        """
        메시지 처리 메인 파이프라인
        
        Args:
            user_text: 사용자 입력
            ai_response: AI 응답 텍스트
            
        Returns:
            dict: video_path, audio_path, duration
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 1. TTS 생성
        tts_path = self.temp_dir / "tts" / f"tts_{timestamp}.wav"
        if not self.tts_processor.create_audio(ai_response, str(tts_path)):
            return {
                'video_path': str(self.base_video),
                'audio_path': None,
                'duration': 3.0
            }
        
        # 2. 오디오 후처리
        audio, sr = librosa.load(str(tts_path), sr=16000, dtype=np.float32)
        audio_path = self.temp_dir / "audio" / f"audio_{timestamp}.wav"
        sf.write(str(audio_path), audio, sr)
        tts_path.unlink(missing_ok=True)
        
        # 3. 립싱크 비디오 생성
        temp_video = self.temp_dir / "videos" / f"temp_{timestamp}.mp4"
        self._generate_lipsync_video(
            str(audio_path),
            str(self.base_video),
            str(temp_video)
        )
        
        # 4. 오디오-비디오 합성
        final_video = self.temp_dir / "videos" / f"response_{timestamp}.mp4"
        if self._merge_audio_video(str(temp_video), str(audio_path), str(final_video)):
            video_path = str(final_video)
            temp_video.unlink(missing_ok=True)
        else:
            video_path = str(self.base_video)
        
        # 5. GPU 메모리 정리
        self.lipsync_model.cleanup()
        
        return {
            'video_path': video_path,
            'audio_path': str(audio_path),
            'duration': len(audio) / sr
        }
    
    def _generate_lipsync_video(self, audio_path: str, source_path: str, output_path: str):
        """Ditto 모델로 립싱크 비디오 생성"""
        # 오디오 로드
        audio, _ = librosa.load(audio_path, sr=16000, dtype=np.float32)
        
        # SDK 설정
        self.sdk.setup(source_path, output_path, **self.speed_config)
        
        # 프레임 수 계산
        audio_duration = len(audio) / 16000
        num_frames = min(int(audio_duration * 20) + 5, 200)
        
        # 프레임 설정
        self.sdk.setup_Nd(
            N_d=num_frames,
            fade_in=-1,
            fade_out=-1,
            ctrl_info={}
        )
        
        # HuBERT로 오디오 특징 추출
        aud_feat = self.sdk.wav2feat.wav2feat(audio, sr=16000)
        
        # 립싱크 생성
        self.sdk.audio2motion_queue.put(aud_feat)
        self.sdk.close()
    
    def _merge_audio_video(self, video_path: str, audio_path: str, output_path: str) -> bool:
        """FFmpeg로 오디오와 비디오 합성"""
        # SDK 임시 비디오 경로
        tmp_video = self.sdk.tmp_output_path if hasattr(self.sdk, 'tmp_output_path') else video_path
        
        # FFmpeg 명령어
        cmd = (
            f'ffmpeg -loglevel error -y '              # 조용히, 덮어쓰기
            f'-hwaccel cuda '                          # GPU 가속
            f'-i "{tmp_video}" '                      # 입력: 생성된 비디오
            f'-i "{audio_path}" '                      # 입력: 원본 오디오
            f'-map 0:v -map 1:a '                      # 비디오는 첫 번째, 오디오는 두 번째
            
            f'-vf "unsharp=5:5:1.5:5:5:0.0" '          # 샤프닝 필터 추가!
            f'-c:v libx264 '                           # 필터 사용시 재인코딩 필수
            f'-preset ultrafast '                      # 가장 빠른 인코딩
            f'-crf 18 '                                # 좋은 품질
            f'-c:a aac '                               # 오디오 AAC 인코딩
            
            # f'-c:v copy -c:a aac '                   # 비디오 복사, 오디오 AAC 인코딩
            f'-af "aresample=async=1:first_pts=0" '    # 오디오 동기화
            f'"{output_path}"'                         # 최종 출력
        )
        
        result = os.system(cmd)
        return result == 0 and Path(output_path).exists()
