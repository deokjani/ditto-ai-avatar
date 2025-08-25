"""
TTS (Text-to-Speech) 처리 모듈
"""
import io
import wave
import base64
import requests
import numpy as np
import soundfile as sf

import v1.config as config


class TTSProcessor:
    """Google TTS를 사용한 음성 생성"""
    
    def __init__(self):
        """초기화"""
        self.api_key = config.TTS_API_KEY
        self.sample_rate = 16000
        
    def create_audio(self, text: str, output_path: str) -> bool:
        """
        텍스트를 음성으로 변환
        
        Args:
            text: 변환할 텍스트
            output_path: 저장 경로
            
        Returns:
            bool: 성공 여부
        """
        try:
            # Google TTS API 호출
            response = requests.post(
                "https://texttospeech.googleapis.com/v1/text:synthesize",
                headers={"X-Goog-Api-Key": self.api_key},
                json={
                    "input": {"text": text},
                    "voice": {
                        "languageCode": "en-US",
                        "name": "en-US-Chirp-HD-F",  # 고품질 여성 음성
                    },
                    "audioConfig": {
                        "audioEncoding": "LINEAR16",
                        "sampleRateHertz": self.sample_rate,
                        "speakingRate": 1.0,
                    }
                }
            )
            
            # Base64 → WAV 변환
            audio_data = base64.b64decode(response.json()['audioContent'])
            audio_array = self._decode_wav(audio_data)
            
            # 무음 패딩 추가
            padded_audio = self._add_silence_padding(audio_array)
            
            # 파일 저장
            sf.write(output_path, padded_audio, self.sample_rate)
            return True
            
        except Exception as e:
            print(f"TTS Error: {e}")
            return False
    
    def _decode_wav(self, audio_data: bytes) -> np.ndarray:
        """WAV 바이트를 numpy 배열로 변환"""
        with io.BytesIO(audio_data) as wav_io:
            with wave.open(wav_io, 'rb') as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    
    def _add_silence_padding(self, audio: np.ndarray, front_sec: float = 0.2, back_sec: float = 0.5) -> np.ndarray:
        """오디오 앞뒤 무음 추가"""
        silence_front = np.zeros(int(self.sample_rate * front_sec), dtype=np.float32)
        silence_back = np.zeros(int(self.sample_rate * back_sec), dtype=np.float32)
        return np.concatenate([silence_front, audio, silence_back])
