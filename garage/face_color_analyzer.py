#!/usr/bin/env python3
"""
얼굴 영역만 색상 분석하는 도구
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime

class FaceColorAnalyzer:
    def __init__(self):
        self.project_root = Path(__file__).parent
        
    def extract_face_region(self, frame):
        """얼굴 영역만 추출 (중앙 부분)"""
        h, w = frame.shape[:2]
        # 중앙 40% 영역만 추출 (얼굴이 있을 가능성이 높은 부분)
        x1 = int(w * 0.3)
        x2 = int(w * 0.7)
        y1 = int(h * 0.2)
        y2 = int(h * 0.6)
        return frame[y1:y2, x1:x2]
    
    def analyze_skin_tone(self, video_path, frame_num=10):
        """피부톤 분석"""
        cap = cv2.VideoCapture(str(video_path))
        
        # 특정 프레임으로 이동
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            print(f"❌ 프레임 읽기 실패: {video_path}")
            return None
            
        # BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 얼굴 영역 추출
        face_region = self.extract_face_region(frame_rgb)
        
        # 피부색 범위 마스크 (HSV)
        face_hsv = cv2.cvtColor(face_region, cv2.COLOR_RGB2HSV)
        
        # 피부색 범위 (조정 가능)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # 마스크 생성
        skin_mask = cv2.inRange(face_hsv, lower_skin, upper_skin)
        
        # 피부 영역만 추출
        skin_pixels = face_region[skin_mask > 0]
        
        if len(skin_pixels) == 0:
            print("⚠️ 피부 영역을 찾을 수 없습니다")
            return None
            
        # RGB 평균값 계산
        mean_rgb = np.mean(skin_pixels, axis=0)
        
        # LAB 색공간으로 변환 (더 정확한 색상 비교)
        skin_pixels_lab = cv2.cvtColor(
            skin_pixels.reshape(-1, 1, 3).astype(np.uint8), 
            cv2.COLOR_RGB2LAB
        )
        mean_lab = np.mean(skin_pixels_lab.reshape(-1, 3), axis=0)
        
        cap.release()
        
        return {
            'rgb': mean_rgb.tolist(),
            'lab': mean_lab.tolist(),
            'frame': frame_rgb,
            'face_region': face_region,
            'skin_pixels': len(skin_pixels)
        }
    
    def compare_videos(self, base_video, result_video):
        """두 비디오의 피부톤 비교"""
        print("\n" + "="*60)
        print("🔍 얼굴 영역 색상 분석")
        print("="*60)
        
        # 여러 프레임 분석
        frame_numbers = [5, 10, 15, 20, 25]
        
        base_colors = []
        result_colors = []
        
        for frame_num in frame_numbers:
            print(f"\n📊 프레임 {frame_num} 분석 중...")
            
            base_data = self.analyze_skin_tone(base_video, frame_num)
            result_data = self.analyze_skin_tone(result_video, frame_num)
            
            if base_data and result_data:
                base_colors.append(base_data)
                result_colors.append(result_data)
                
                # RGB 차이
                rgb_diff = np.array(result_data['rgb']) - np.array(base_data['rgb'])
                print(f"  RGB 차이: R={rgb_diff[0]:.1f}, G={rgb_diff[1]:.1f}, B={rgb_diff[2]:.1f}")
                
                # LAB 차이 (색상 지각 차이)
                lab_diff = np.array(result_data['lab']) - np.array(base_data['lab'])
                print(f"  LAB 차이: L={lab_diff[0]:.1f}, a={lab_diff[1]:.1f}, b={lab_diff[2]:.1f}")
                
                # Delta E (색상 차이 지표)
                delta_e = np.sqrt(np.sum(lab_diff**2))
                print(f"  Delta E: {delta_e:.2f}")
                
                if delta_e < 2:
                    print("  → 거의 구분 불가")
                elif delta_e < 5:
                    print("  → 약간의 차이")
                elif delta_e < 10:
                    print("  → 눈에 띄는 차이")
                else:
                    print("  → 큰 차이")
        
        # 평균 계산
        if base_colors and result_colors:
            avg_base_rgb = np.mean([c['rgb'] for c in base_colors], axis=0)
            avg_result_rgb = np.mean([c['rgb'] for c in result_colors], axis=0)
            avg_rgb_diff = avg_result_rgb - avg_base_rgb
            
            avg_base_lab = np.mean([c['lab'] for c in base_colors], axis=0)
            avg_result_lab = np.mean([c['lab'] for c in result_colors], axis=0)
            avg_lab_diff = avg_result_lab - avg_base_lab
            avg_delta_e = np.sqrt(np.sum(avg_lab_diff**2))
            
            print("\n" + "="*60)
            print("📈 전체 평균 분석 결과")
            print("="*60)
            print(f"\n🎨 평균 RGB 차이:")
            print(f"  R: {avg_rgb_diff[0]:+.1f} {'(더 붉음)' if avg_rgb_diff[0] > 0 else '(덜 붉음)'}")
            print(f"  G: {avg_rgb_diff[1]:+.1f}")
            print(f"  B: {avg_rgb_diff[2]:+.1f}")
            
            print(f"\n🌈 평균 LAB 차이:")
            print(f"  L (밝기): {avg_lab_diff[0]:+.1f}")
            print(f"  a (빨강-초록): {avg_lab_diff[1]:+.1f} {'(더 붉은 톤)' if avg_lab_diff[1] > 0 else '(더 초록 톤)'}")
            print(f"  b (노랑-파랑): {avg_lab_diff[2]:+.1f} {'(더 노란 톤)' if avg_lab_diff[2] > 0 else '(더 파란 톤)'}")
            
            print(f"\n📏 평균 Delta E: {avg_delta_e:.2f}")
            
            # 시각화
            self.visualize_comparison(base_colors[0], result_colors[0])
            
            return {
                'avg_rgb_diff': avg_rgb_diff.tolist(),
                'avg_lab_diff': avg_lab_diff.tolist(),
                'avg_delta_e': float(avg_delta_e)
            }
    
    def visualize_comparison(self, base_data, result_data):
        """시각적 비교"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 원본 프레임
        axes[0, 0].imshow(base_data['frame'])
        axes[0, 0].set_title('원본 전체')
        axes[0, 0].axis('off')
        
        # 원본 얼굴 영역
        axes[0, 1].imshow(base_data['face_region'])
        axes[0, 1].set_title('원본 얼굴 영역')
        axes[0, 1].axis('off')
        
        # 원본 색상 패치
        color_patch_base = np.full((100, 100, 3), base_data['rgb'], dtype=np.uint8)
        axes[0, 2].imshow(color_patch_base)
        axes[0, 2].set_title(f"원본 피부톤\nRGB: {[int(v) for v in base_data['rgb']]}")
        axes[0, 2].axis('off')
        
        # 생성된 프레임
        axes[1, 0].imshow(result_data['frame'])
        axes[1, 0].set_title('생성된 비디오 전체')
        axes[1, 0].axis('off')
        
        # 생성된 얼굴 영역
        axes[1, 1].imshow(result_data['face_region'])
        axes[1, 1].set_title('생성된 비디오 얼굴 영역')
        axes[1, 1].axis('off')
        
        # 생성된 색상 패치
        color_patch_result = np.full((100, 100, 3), result_data['rgb'], dtype=np.uint8)
        axes[1, 2].imshow(color_patch_result)
        axes[1, 2].set_title(f"생성된 피부톤\nRGB: {[int(v) for v in result_data['rgb']]}")
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # 저장
        output_dir = self.project_root / "color_analysis"
        output_dir.mkdir(exist_ok=True)
        
        output_path = output_dir / "face_color_comparison.png"
        plt.savefig(output_path, dpi=150)
        print(f"\n✅ 얼굴 색상 비교 저장: {output_path}")
        
        plt.show()

def main():
    analyzer = FaceColorAnalyzer()
    
    # 파일 경로
    base_video = Path("example/base.mp4")
    result_video = Path("tmp/result_20250822_100105.mp4")  # 최신 결과 파일로 변경
    
    if not base_video.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {base_video}")
        return
        
    if not result_video.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {result_video}")
        return
    
    # 분석 실행
    results = analyzer.compare_videos(str(base_video), str(result_video))
    
    # 결과 저장
    if results:
        output_dir = Path("color_analysis")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"face_color_analysis_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 분석 결과 저장: {output_file}")

if __name__ == "__main__":
    main()
