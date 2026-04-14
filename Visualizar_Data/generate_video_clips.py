#!/usr/bin/env python3
"""
Generate video clips of normal vs suspicious people for visualization.

Creates side-by-side and grid-based videos showing:
- Multiple normal shopping pose sequences
- Multiple suspicious/shoplifting pose sequences
- Color-coded skeletons with smooth animations
- Configurable number of clips via command-line arguments

Usage:
    python generate_video_clips.py --normal 5 --suspicious 5
    python generate_video_clips.py --normal 10 --suspicious 10 --fps 20
    python generate_video_clips.py --help

Output:
    Output/
    ├── normal_clips_grid.mp4 (all normal clips in grid)
    ├── suspicious_clips_grid.mp4 (all suspicious clips in grid)
    ├── normal_vs_suspicious.mp4 (alternating comparison)
    └── individual_clips/
        ├── normal_clip_001.mp4
        ├── normal_clip_002.mp4
        ├── suspicious_clip_001.mp4
        └── ...
"""

import json
import numpy as np
from pathlib import Path
import cv2
import argparse
import random
from tqdm import tqdm
from visualize import draw_skeleton, rotate_keypoints_90_ccw

# ==================== Configuration ====================

OUTPUT_DIR = Path(__file__).parent / 'Output'
MANIFEST_PATH = Path(__file__).parent / 'Suspicious_Dataset' / 'manifest.json'
SEQUENCES_DIR = Path(__file__).parent / 'Suspicious_Dataset' / 'person_sequences'

CANVAS_HEIGHT = 1400
CANVAS_WIDTH = 800
DEFAULT_FPS = 24
DEFAULT_FRAME_DELAY = 1  # Show each frame for N frames in video (slower = clearer)

# Video codec
FOURCC = cv2.VideoWriter_fourcc(*'mp4v')

# Colors
COLOR_NORMAL = (0, 200, 0)  # Green
COLOR_SUSPICIOUS = (0, 0, 255)  # Red
COLOR_BACKGROUND = np.uint8(220)


class VideoClipGenerator:
    """Generate video clips from skeleton pose sequences."""
    
    def __init__(self, manifest_path, sequences_dir, output_dir, fps=DEFAULT_FPS):
        """Initialize the generator."""
        self.manifest_path = manifest_path
        self.sequences_dir = sequences_dir
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.frame_delay = DEFAULT_FRAME_DELAY
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / 'individual_clips').mkdir(exist_ok=True)
        
        # Load manifest
        print("📂 Loading manifest...")
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        print(f"   Loaded {len(self.manifest)} people")
        
        # Split by label
        self.normal_people = [e for e in self.manifest if e['dataset'] == 'Real-world']
        self.suspicious_people = [e for e in self.manifest if e['dataset'] == 'Staged']
        
        print(f"   Normal (Real-world): {len(self.normal_people)}")
        print(f"   Suspicious (Staged): {len(self.suspicious_people)}")
    
    def load_person_frames(self, person_id):
        """Load all frames for a person."""
        person_dir = self.sequences_dir / person_id
        if not person_dir.exists():
            return None
        
        frames_data = []
        frame_files = sorted(list(person_dir.glob('*.json')))
        
        for frame_file in frame_files:
            try:
                with open(frame_file) as f:
                    data = json.load(f)
                frames_data.append({
                    'frame_id': data['frame_id'],
                    'keypoints': data['keypoints'],
                    'confidence': data.get('confidence', 0.5)
                })
            except Exception as e:
                print(f"   ⚠ Error loading {frame_file}: {e}")
                continue
        
        return frames_data if frames_data else None
    
    def render_frame(self, keypoints, label, label_text):
        """Render a single frame with skeleton."""
        img = np.ones((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8) * COLOR_BACKGROUND
        
        color = COLOR_NORMAL if label == 'normal' else COLOR_SUSPICIOUS
        
        try:
            draw_skeleton(
                img, keypoints,
                color=color,
                confidence_threshold=0.2,
                thickness=3, kpt_size=5,
                draw_connections=True,
                max_limb_ratio=0.55
            )
        except Exception as e:
            print(f"   ⚠ Error drawing skeleton: {e}")
            return None
        
        # Add label
        cv2.putText(img, label_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        return img
    
    def render_grid(self, frames_list, label, label_text, grid_cols=2):
        """Render multiple frames in a grid pattern."""
        """Returns list of frames, one per time step across all frames."""
        
        if not frames_list:
            return []
        
        # Convert any numpy arrays to lists
        frames_list = [list(f) if isinstance(f, np.ndarray) else f for f in frames_list]
        
        # Find max number of frames any clip has
        max_frames = max(len(f) for f in frames_list)
        grid_rows = (len(frames_list) + grid_cols - 1) // grid_cols
        
        grid_height = CANVAS_HEIGHT * grid_rows
        grid_width = CANVAS_WIDTH * grid_cols
        canvas_frames = []
        
        # For each time step
        for t in range(max_frames):
            canvas = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * int(COLOR_BACKGROUND)
            
            # For each clip in the grid
            for idx, frame_sequence in enumerate(frames_list):
                row = idx // grid_cols
                col = idx % grid_cols
                
                y_start = row * CANVAS_HEIGHT
                x_start = col * CANVAS_WIDTH
                y_end = y_start + CANVAS_HEIGHT
                x_end = x_start + CANVAS_WIDTH
                
                # Get frame at time t (or last frame if sequence is shorter)
                frame_idx = min(t, len(frame_sequence) - 1)
                frame_data = frame_sequence[frame_idx]
                
                # Render skeleton
                frame_img = self.render_frame(frame_data, label, f"{label_text}")
                if frame_img is not None:
                    canvas[y_start:y_end, x_start:x_end] = frame_img
                
                # Add clip number
                cv2.putText(canvas, f"Clip {idx+1}", 
                           (x_start + 20, y_start + CANVAS_HEIGHT - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            
            # Repeat frame for smooth playback
            for _ in range(self.frame_delay):
                canvas_frames.append(canvas)
        
        return canvas_frames
    
    def generate_individual_clips(self, people_list, label, num_clips):
        """Generate individual video clips."""
        print(f"\n🎬 Generating individual {label} clips...")
        
        clip_videos = []
        
        for i, person_entry in enumerate(tqdm(people_list[:num_clips], desc=f"  {label} clips")):
            person_id = person_entry['person_id']
            
            # Load frames
            frames_data = self.load_person_frames(person_id)
            if not frames_data or len(frames_data) < 2:
                continue
            
            # Render frames
            rendered_frames = []
            for frame_data in frames_data:
                img = self.render_frame(frame_data['keypoints'], label,
                                       f"{label.upper()} - {person_id}")
                if img is not None:
                    # Repeat for smooth playback
                    for _ in range(self.frame_delay):
                        rendered_frames.append(img)
            
            if not rendered_frames:
                continue
            
            # Save individual clip
            clip_num = i + 1
            output_file = self.output_dir / 'individual_clips' / f'{label}_clip_{clip_num:03d}.mp4'
            self._write_video(output_file, rendered_frames)
            clip_videos.append(rendered_frames)
        
        return clip_videos
    
    def _write_video(self, output_path, frames):
        """Write frames to video file."""
        if not frames:
            return False
        
        h, w = frames[0].shape[:2]
        
        try:
            writer = cv2.VideoWriter(str(output_path), FOURCC, self.fps, (w, h))
            for frame in frames:
                writer.write(frame)
            writer.release()
            print(f"   ✅ Saved: {output_path.name}")
            return True
        except Exception as e:
            print(f"   ❌ Error writing {output_path.name}: {e}")
            return False
    

    def generate_all(self, num_normal, num_suspicious, fps=None):
        """Generate all videos."""
        if fps:
            self.fps = fps
        
        print("\n" + "="*70)
        print("🎬 VIDEO CLIP GENERATION")
        print("="*70)
        print(f"FPS: {self.fps}")
        print(f"Normal clips: {num_normal}")
        print(f"Suspicious clips: {num_suspicious}")
        
        # Select random people
        selected_normal = random.sample(self.normal_people, 
                                       min(num_normal, len(self.normal_people)))
        selected_suspicious = random.sample(self.suspicious_people,
                                           min(num_suspicious, len(self.suspicious_people)))
        
        # Generate individual clips
        normal_clips_data = self.generate_individual_clips(selected_normal, 'normal', num_normal)
        suspicious_clips_data = self.generate_individual_clips(selected_suspicious, 'suspicious', num_suspicious)
        
        # Generate grid videos (load raw data)
        if selected_normal:
            normal_raw = [self.load_person_frames(p['person_id']) for p in selected_normal[:num_normal]]
            normal_raw = [f for f in normal_raw if f is not None]
            if normal_raw:
                self.generate_grid_from_raw(normal_raw, 'normal', 'normal_clips_grid.mp4')
        
        if selected_suspicious:
            suspicious_raw = [self.load_person_frames(p['person_id']) for p in selected_suspicious[:num_suspicious]]
            suspicious_raw = [f for f in suspicious_raw if f is not None]
            if suspicious_raw:
                self.generate_grid_from_raw(suspicious_raw, 'suspicious', 'suspicious_clips_grid.mp4')
        
        # Generate comparison (load raw data)
        if selected_normal and selected_suspicious:
            normal_raw = [self.load_person_frames(p['person_id']) for p in selected_normal[:num_normal]]
            suspicious_raw = [self.load_person_frames(p['person_id']) for p in selected_suspicious[:num_suspicious]]
            normal_raw = [f for f in normal_raw if f is not None]
            suspicious_raw = [f for f in suspicious_raw if f is not None]
            if normal_raw and suspicious_raw:
                self.generate_comparison_from_raw(normal_raw, suspicious_raw, 
                                                 'normal_vs_suspicious.mp4')
        
        print("\n" + "="*70)
        print("✅ COMPLETE!")
        print("="*70)
        print(f"Output files in: {self.output_dir}/")
        print(f"  - individual_clips/ (all clips separately)")
        print(f"  - normal_clips_grid.mp4 (normal clips in grid)")
        print(f"  - suspicious_clips_grid.mp4 (suspicious clips in grid)")
        print(f"  - normal_vs_suspicious.mp4 (side-by-side comparison)")
    
    def generate_grid_from_raw(self, frame_sequences, label, output_name, grid_cols=2):
        """Generate grid video from raw frame data."""
        print(f"\n📊 Generating {label} grid video...")
        
        if not frame_sequences:
            print(f"   ❌ No clips to grid")
            return False
        
        # Find max frames
        max_frames = max(len(f) for f in frame_sequences)
        grid_rows = (len(frame_sequences) + grid_cols - 1) // grid_cols
        
        grid_height = CANVAS_HEIGHT * grid_rows
        grid_width = CANVAS_WIDTH * grid_cols
        canvas_frames = []
        
        # For each time step
        for t in range(max_frames):
            canvas = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * int(COLOR_BACKGROUND)
            
            for idx, sequence in enumerate(frame_sequences):
                row = idx // grid_cols
                col = idx % grid_cols
                
                y_start = row * CANVAS_HEIGHT
                x_start = col * CANVAS_WIDTH
                y_end = y_start + CANVAS_HEIGHT
                x_end = x_start + CANVAS_WIDTH
                
                # Get frame at time t
                frame_idx = min(t, len(sequence) - 1)
                frame_data = sequence[frame_idx]
                
                # Render skeleton
                frame_img = self.render_frame(frame_data['keypoints'], label, f"{label.upper()}")
                if frame_img is not None:
                    canvas[y_start:y_end, x_start:x_end] = frame_img
                
                cv2.putText(canvas, f"Clip {idx+1}", 
                           (x_start + 20, y_start + CANVAS_HEIGHT - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            
            # Repeat frame
            for _ in range(self.frame_delay):
                canvas_frames.append(canvas)
        
        if canvas_frames:
            output_file = self.output_dir / output_name
            return self._write_video(output_file, canvas_frames)
        
        return False
    
    def generate_comparison_from_raw(self, normal_sequences, suspicious_sequences, output_name):
        """Generate comparison video from raw frame data."""
        print(f"\n🔄 Generating comparison video...")
        
        if not normal_sequences or not suspicious_sequences:
            print(f"   ❌ Need both normal and suspicious clips")
            return False
        
        min_clips = min(len(normal_sequences), len(suspicious_sequences))
        comparison_frames = []
        
        for pair_idx in range(min_clips):
            normal_seq = normal_sequences[pair_idx]
            suspicious_seq = suspicious_sequences[pair_idx]
            
            max_f = max(len(normal_seq), len(suspicious_seq))
            
            for t in range(max_f):
                canvas = np.ones((CANVAS_HEIGHT, CANVAS_WIDTH * 2, 3), dtype=np.uint8) * int(COLOR_BACKGROUND)
                
                # Left: normal
                frame_idx = min(t, len(normal_seq) - 1)
                frame_img = self.render_frame(normal_seq[frame_idx]['keypoints'], 'normal', 'NORMAL')
                if frame_img is not None:
                    canvas[:, :CANVAS_WIDTH] = frame_img
                
                # Right: suspicious
                frame_idx = min(t, len(suspicious_seq) - 1)
                frame_img = self.render_frame(suspicious_seq[frame_idx]['keypoints'], 'suspicious', 'SUSPICIOUS')
                if frame_img is not None:
                    canvas[:, CANVAS_WIDTH:] = frame_img
                
                # Title
                cv2.putText(canvas, "NORMAL", (40, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLOR_NORMAL, 3)
                cv2.putText(canvas, "SUSPICIOUS", (CANVAS_WIDTH + 40, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLOR_SUSPICIOUS, 3)
                
                comparison_frames.append(canvas)
        
        if comparison_frames:
            output_file = self.output_dir / output_name
            return self._write_video(output_file, comparison_frames)
        
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate video clips of normal vs suspicious pose sequences',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python generate_video_clips.py --normal 5 --suspicious 5
  python generate_video_clips.py --normal 10 --suspicious 10 --fps 20
  python generate_video_clips.py --normal 3 --suspicious 3 --fps 30
        '''
    )
    
    parser.add_argument('--normal', type=int, default=5,
                       help='Number of normal clips to generate (default: 5)')
    parser.add_argument('--suspicious', type=int, default=5,
                       help='Number of suspicious clips to generate (default: 5)')
    parser.add_argument('--fps', type=int, default=DEFAULT_FPS,
                       help=f'Video FPS (default: {DEFAULT_FPS})')
    parser.add_argument('--output', type=str, default='Output',
                       help='Output directory (default: Output)')
    
    args = parser.parse_args()
    
    # Verify manifest exists
    if not MANIFEST_PATH.exists():
        print("❌ Error: Suspicious_Dataset/manifest.json not found!")
        print("   Run: python build_suspicious_manifest.py")
        return False
    
    # Generate videos
    try:
        generator = VideoClipGenerator(
            MANIFEST_PATH,
            SEQUENCES_DIR,
            args.output,
            fps=args.fps
        )
        generator.generate_all(args.normal, args.suspicious)
        return True
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
