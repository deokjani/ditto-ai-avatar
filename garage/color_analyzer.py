#!/usr/bin/env python3
"""
Ditto TalkingHead 색상 비교 분석 도구
원본 base.mp4와 생성된 비디오의 색상을 비교 분석합니다.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import Tuple, Dict, List
import json
from datetime import datetime

class DittoColorAnalyzer:
    def __init__(self, project_root: str = "/data/edutem/deokjin/ai-human/ditto-talkinghead"):
        """
        초기화
        Args:
            project_root: Ditto 프로젝트 루트 경로
        """
        self.project_root = Path(project_root)
        self.base_video = self.project_root / "example" / "base.mp4"
        self.output_dir = self.project_root / "color_analysis"
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"🎨 Ditto Color Analyzer 초기화")
        print(f"📁 프로젝트 경로: {self.project_root}")
        print(f"📹 기준 비디오: {self.base_video}")
    
    def extract_frames(self, video_path: str, num_frames: int = 5) -> List[np.ndarray]:
        """
        비디오에서 여러 프레임 추출
        
        Args:
            video_path: 비디오 파일 경로
            num_frames: 추출할 프레임 수
        
        Returns:
            프레임 리스트 (RGB 형식)
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            raise ValueError(f"비디오를 읽을 수 없습니다: {video_path}")
        
        # 균등 간격으로 프레임 선택
        frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
        frames = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        print(f"✅ {len(frames)}개 프레임 추출 완료: {Path(video_path).name}")
        return frames
    
    def calculate_color_stats(self, image: np.ndarray) -> Dict:
        """
        이미지의 색상 통계 계산
        
        Args:
            image: 입력 이미지 (RGB)
        
        Returns:
            색상 통계 딕셔너리
        """
        stats = {}
        
        # RGB 통계
        for i, channel in enumerate(['R', 'G', 'B']):
            channel_data = image[:, :, i]
            stats[f'{channel}_mean'] = float(np.mean(channel_data))
            stats[f'{channel}_std'] = float(np.std(channel_data))
            stats[f'{channel}_min'] = float(np.min(channel_data))
            stats[f'{channel}_max'] = float(np.max(channel_data))
        
        # HSV 통계
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        for i, channel in enumerate(['H', 'S', 'V']):
            channel_data = hsv[:, :, i]
            stats[f'{channel}_mean'] = float(np.mean(channel_data))
            stats[f'{channel}_std'] = float(np.std(channel_data))
        
        # 밝기 (그레이스케일)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        stats['brightness_mean'] = float(np.mean(gray))
        stats['brightness_std'] = float(np.std(gray))
        
        return stats
    
    def calculate_histogram(self, image: np.ndarray, bins: int = 256) -> Dict:
        """
        히스토그램 계산
        
        Args:
            image: 입력 이미지 (RGB)
            bins: 히스토그램 빈 수
        
        Returns:
            각 채널의 히스토그램
        """
        histograms = {}
        
        # RGB 히스토그램
        for i, channel in enumerate(['R', 'G', 'B']):
            hist, _ = np.histogram(image[:, :, i], bins=bins, range=(0, 256))
            histograms[channel] = hist
        
        # 밝기 히스토그램
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hist, _ = np.histogram(gray, bins=bins, range=(0, 256))
        histograms['brightness'] = hist
        
        return histograms
    
    def compare_videos(self, generated_video_path: str, num_frames: int = 5):
        """
        원본과 생성된 비디오 비교
        
        Args:
            generated_video_path: 생성된 비디오 경로
            num_frames: 비교할 프레임 수
        """
        print("\n" + "="*60)
        print("🔍 색상 비교 분석 시작")
        print("="*60)
        
        # 프레임 추출
        original_frames = self.extract_frames(str(self.base_video), num_frames)
        generated_frames = self.extract_frames(generated_video_path, num_frames)
        
        # 전체 통계
        all_stats = {
            'original': [],
            'generated': [],
            'differences': []
        }
        
        # 각 프레임 쌍 비교
        for i, (orig, gen) in enumerate(zip(original_frames, generated_frames)):
            print(f"\n📊 프레임 {i+1}/{num_frames} 분석 중...")
            
            # 통계 계산
            orig_stats = self.calculate_color_stats(orig)
            gen_stats = self.calculate_color_stats(gen)
            
            # 차이 계산
            diff_stats = {}
            for key in orig_stats:
                diff_stats[key] = gen_stats[key] - orig_stats[key]
            
            all_stats['original'].append(orig_stats)
            all_stats['generated'].append(gen_stats)
            all_stats['differences'].append(diff_stats)
        
        # 결과 출력
        self._print_analysis_results(all_stats)
        
        # 시각화
        self._create_visualizations(original_frames, generated_frames, all_stats)
        
        # JSON으로 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = self.output_dir / f"color_analysis_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_stats, f, indent=2, ensure_ascii=False)
        print(f"\n💾 분석 결과 저장: {json_path}")
    
    def _print_analysis_results(self, stats: Dict):
        """분석 결과 출력"""
        print("\n" + "="*60)
        print("📈 색상 변화 분석 결과")
        print("="*60)
        
        # 평균 차이 계산
        avg_diff = {}
        for diff in stats['differences']:
            for key, value in diff.items():
                if key not in avg_diff:
                    avg_diff[key] = []
                avg_diff[key].append(value)
        
        for key in avg_diff:
            avg_diff[key] = np.mean(avg_diff[key])
        
        # RGB 채널별 변화
        print("\n🎨 RGB 채널 평균 변화:")
        for channel in ['R', 'G', 'B']:
            mean_diff = avg_diff[f'{channel}_mean']
            std_diff = avg_diff[f'{channel}_std']
            print(f"  {channel}: 평균 {mean_diff:+.2f}, 표준편차 {std_diff:+.2f}")
        
        # HSV 변화
        print("\n🌈 HSV 채널 평균 변화:")
        print(f"  H (색상): {avg_diff['H_mean']:+.2f}")
        print(f"  S (채도): {avg_diff['S_mean']:+.2f}")
        print(f"  V (명도): {avg_diff['V_mean']:+.2f}")
        
        # 밝기 변화
        print(f"\n💡 전체 밝기 변화: {avg_diff['brightness_mean']:+.2f}")
        
        # 색상 편향 진단
        print("\n🔍 진단:")
        rgb_shifts = [avg_diff['R_mean'], avg_diff['G_mean'], avg_diff['B_mean']]
        max_shift = max(abs(s) for s in rgb_shifts)
        
        if max_shift > 20:
            print("  ⚠️ 심각한 색상 변화 감지됨!")
        elif max_shift > 10:
            print("  ⚡ 중간 정도의 색상 변화 감지됨")
        elif max_shift > 5:
            print("  📌 경미한 색상 변화 감지됨")
        else:
            print("  ✅ 색상이 잘 유지되고 있습니다")
        
        # 가장 큰 변화
        dominant_shift = max(enumerate(rgb_shifts), key=lambda x: abs(x[1]))
        channels = ['빨강', '초록', '파랑']
        if abs(dominant_shift[1]) > 5:
            direction = "증가" if dominant_shift[1] > 0 else "감소"
            print(f"  → {channels[dominant_shift[0]]} 채널이 가장 크게 {direction} ({dominant_shift[1]:+.2f})")
    
    def _create_visualizations(self, original_frames: List, generated_frames: List, stats: Dict):
        """시각화 생성"""
        print("\n📊 시각화 생성 중...")
        
        # 1. 프레임 비교
        fig, axes = plt.subplots(2, len(original_frames), figsize=(15, 6))
        fig.suptitle('원본 vs 생성된 비디오 프레임 비교', fontsize=16)
        
        for i, (orig, gen) in enumerate(zip(original_frames, generated_frames)):
            axes[0, i].imshow(orig)
            axes[0, i].set_title(f'원본 프레임 {i+1}')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(gen)
            axes[1, i].set_title(f'생성 프레임 {i+1}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        frame_comp_path = self.output_dir / "frame_comparison.png"
        plt.savefig(frame_comp_path, dpi=150)
        plt.close()
        
        # 2. 히스토그램 비교 (첫 프레임)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('색상 히스토그램 비교 (첫 프레임)', fontsize=16)
        
        orig_hist = self.calculate_histogram(original_frames[0])
        gen_hist = self.calculate_histogram(generated_frames[0])
        
        # RGB 히스토그램
        colors = ['red', 'green', 'blue']
        for i, (channel, color) in enumerate(zip(['R', 'G', 'B'], colors)):
            axes[0, 0].plot(orig_hist[channel], color=color, alpha=0.7, label=f'{channel} (원본)')
            axes[0, 0].plot(gen_hist[channel], color=color, alpha=0.7, linestyle='--', label=f'{channel} (생성)')
        axes[0, 0].set_title('RGB 히스토그램')
        axes[0, 0].set_xlabel('픽셀 값')
        axes[0, 0].set_ylabel('빈도')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 밝기 히스토그램
        axes[0, 1].plot(orig_hist['brightness'], color='black', alpha=0.7, label='원본')
        axes[0, 1].plot(gen_hist['brightness'], color='black', alpha=0.7, linestyle='--', label='생성')
        axes[0, 1].set_title('밝기 히스토그램')
        axes[0, 1].set_xlabel('픽셀 값')
        axes[0, 1].set_ylabel('빈도')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 색상 차이 맵
        diff_image = np.abs(original_frames[0].astype(float) - generated_frames[0].astype(float))
        axes[1, 0].imshow(diff_image.astype(np.uint8))
        axes[1, 0].set_title('절대 차이 맵')
        axes[1, 0].axis('off')
        
        # 차이 히트맵 (그레이스케일)
        gray_diff = cv2.cvtColor(diff_image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        im = axes[1, 1].imshow(gray_diff, cmap='hot')
        axes[1, 1].set_title('차이 히트맵')
        axes[1, 1].axis('off')
        plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        hist_comp_path = self.output_dir / "histogram_comparison.png"
        plt.savefig(hist_comp_path, dpi=150)
        plt.close()
        
        print(f"✅ 프레임 비교 저장: {frame_comp_path}")
        print(f"✅ 히스토그램 비교 저장: {hist_comp_path}")
    
    def analyze_latest_output(self):
        """가장 최근 생성된 비디오 분석"""
        # tmp 폴더에서 가장 최근 mp4 파일 찾기
        tmp_dir = self.project_root / "tmp"
        if not tmp_dir.exists():
            print("❌ tmp 폴더를 찾을 수 없습니다.")
            return
        
        mp4_files = list(tmp_dir.glob("*.mp4"))
        if not mp4_files:
            print("❌ 생성된 비디오 파일을 찾을 수 없습니다.")
            return
        
        # 가장 최근 파일
        latest_video = max(mp4_files, key=lambda p: p.stat().st_mtime)
        print(f"🎬 분석할 비디오: {latest_video}")
        
        self.compare_videos(str(latest_video))


def main():
    parser = argparse.ArgumentParser(description="Ditto TalkingHead 색상 비교 분석")
    parser.add_argument(
        "--project-root",
        type=str,
        default="/data/edutem/deokjin/ai-human/ditto-talkinghead",
        help="Ditto 프로젝트 루트 경로"
    )
    parser.add_argument(
        "--video",
        type=str,
        help="분석할 생성된 비디오 경로 (지정하지 않으면 최근 파일 자동 선택)"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=5,
        help="비교할 프레임 수 (기본값: 5)"
    )
    
    args = parser.parse_args()
    
    # 분석기 초기화
    analyzer = DittoColorAnalyzer(args.project_root)
    
    # 분석 실행
    if args.video:
        analyzer.compare_videos(args.video, args.frames)
    else:
        analyzer.analyze_latest_output()
    
    print("\n✨ 분석 완료!")


if __name__ == "__main__":
    main()