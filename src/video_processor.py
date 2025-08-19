"""
비디오 처리
"""
import io
import wave
import time
import base64
import requests
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

import params
from model_pool import DittoModelPool

class VideoProcessor:
    def __init__(self):
        # 경로 설정
        self.project_root = Path(__file__).parent.parent
        self.base_video = self.project_root / "example" / "base.mp4"
        self.temp_dir = Path.home() / ".cache" / "ditto_temp"
        
        # 디렉토리 생성
        (self.temp_dir / "videos").mkdir(parents=True, exist_ok=True)
        (self.temp_dir / "audio").mkdir(parents=True, exist_ok=True)
        (self.temp_dir / "tts").mkdir(parents=True, exist_ok=True)
        
        # 모델 초기화
        self.model_pool = DittoModelPool()
        self._warmup()
    
    def _warmup(self):
        """워밍업"""
        try:
            warmup_audio = np.zeros(16000, dtype=np.float32)
            warmup_path = self.temp_dir / "warmup.wav"
            sf.write(str(warmup_path), warmup_audio, 16000)
            
            warmup_output = self.temp_dir / "warmup.mp4"
            self.model_pool.process(
                str(warmup_path),
                str(self.base_video),
                str(warmup_output)
            )
            
            warmup_path.unlink(missing_ok=True)
            warmup_output.unlink(missing_ok=True)
            # Model warmup completed
        except:
            pass
    
    def google_tts(self, text: str, output_path: str):
        """TTS 생성 - RAM 디스크에 직접 저장"""
        try:
            response = requests.post(
                "https://texttospeech.googleapis.com/v1/text:synthesize",
                headers={"X-Goog-Api-Key": params.TTS_API_KEY},
                json={
                    "input": {"text": text},
                    "voice": {
                        "languageCode": "en-US",
                        "name": "en-US-Chirp-HD-F", 
                    },
                    "audioConfig": {
                        "audioEncoding": "LINEAR16",
                        "sampleRateHertz": 16000,
                        "speakingRate": 1.0,
                    }
                }
            )
            
            audio_data = base64.b64decode(response.json()['audioContent'])
            
            # 메모리에서 처리
            with io.BytesIO(audio_data) as wav_io:
                with wave.open(wav_io, 'rb') as wav_file:
                    frames = wav_file.readframes(wav_file.getnframes())
                    audio_array = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            
            # 패딩 추가
            silence_front = np.zeros(int(16000 * 0.3), dtype=np.float32)
            silence_back = np.zeros(int(16000 * 0.2), dtype=np.float32)
            padded_audio = np.concatenate([silence_front, audio_array, silence_back])
            
            # RAM 디스크에 저장
            sf.write(output_path, padded_audio, 16000)
            return True
            
        except Exception as e:
            # TTS error (조용히 처리)
            pass
            return False
    
    def process_message_with_audio(self, user_text: str, ai_response: str):
        """메시지 처리"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # TTS 생성
        tts_path = self.temp_dir / "tts" / f"tts_{timestamp}.wav"
        if not self.google_tts(ai_response, str(tts_path)):
            return {
                'video_path': str(self.base_video),
                'audio_path': None,
                'duration': 3.0
            }
        
        # 오디오 처리
        audio, sr = librosa.load(str(tts_path), sr=16000, dtype=np.float32)
        audio_path = self.temp_dir / "audio" / f"audio_{timestamp}.wav"
        sf.write(str(audio_path), audio, sr)
        tts_path.unlink(missing_ok=True)
        
        # 비디오 생성
        output_video = self.temp_dir / "videos" / f"response_{timestamp}.mp4"
        video_path = self.model_pool.process(
            str(audio_path),
            str(self.base_video),
            str(output_video)
        ) or str(self.base_video)
        
        # GPU 정리
        self.model_pool.cleanup()
        
        return {
            'video_path': video_path,
            'audio_path': str(audio_path),
            'duration': len(audio) / sr
        }
