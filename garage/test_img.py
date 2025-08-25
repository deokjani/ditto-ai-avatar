import os
from datetime import datetime

print("🚀 Ditto 테스트 시작...")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"./tmp/result_{timestamp}.mp4"

cmd = f"""python inference.py \
    --data_root "./checkpoints/ditto_pytorch" \
    --cfg_pkl "./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl" \
    --audio_path "./example/canyou.mp3" \
    --source_path "./example/base.jpg" \
    --output_path "{output_path}" """

os.system(cmd)
print(f"✅ 완료! 출력 파일: {output_path}")