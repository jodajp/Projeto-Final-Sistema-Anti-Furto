#!/usr/bin/env python
"""
Enhanced video clip generator with:
- Proper nested JSON loading
- Frame interpolation for smooth playback
- Training dataset support
- Much longer sequences
"""

import json
import argparse
import random
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
from scipy.interpolate import interp1d

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
MANIFESTS_DIR = Path(__file__).parent / 'Manifests'
DATA_DIR = Path(__file__).parent / 'Data'

# Video settings
CANVAS_WIDTH, CANVAS_HEIGHT = 1024, 768
COLOR_BACKGROUND = 50
COLOR_NORMAL = (0, 255, 0)  # Green
COLOR_SUSPICIOUS = (0, 0, 255)  # Red
FOURCC = cv2.VideoWriter_fourcc(*'mp4v')

# Default playback
DEFAULT_FPS = 24


def interpolate_keypoints(kp1, kp2, num_frames=3):
    """
    Interpolate between two keypoint frames.
    
    Args:
        kp1, kp2: Lists of 51 values (17 keypoints * 3: x, y, conf)
        num_frames: Number of frames to generate (including endpoints)
    
    Returns:
        List of interpolated keypoints lists
    """
    kp1 = np.array(kp1)
    kp2 = np.array(kp2)
    
    # Create interpolation function
    x = np.array([0, 1])
    y = np.array([kp1, kp2])
    
    # Interpolate
    f = interp1d(x, y.T, kind='cubic', axis=1)
    t = np.linspace(0, 1, num_frames)
    
    interpolated = f(t).T
    return [list(row) for row in interpolated]


def normalize_skeleton_to_frame(keypoints, canvas_w=CANVAS_WIDTH, canvas_h=CANVAS_HEIGHT):
    """
    Normalize skeleton to fit within frame with proper scaling and centering.
    
    Features:
    - Confidence-based filtering
    - Scale-invariant sizing
    - Pelvis-centered positioning (anatomical center)
    - 90-degree rotation (FIX: feet point downward, not rightward)
    - Padding around skeleton bounds
    
    Steps:
    1. Parse keypoints and filter by confidence
    2. Compute bounding box
    3. Apply scale to fit canvas with padding
    4. Center in frame
    5. Apply 90° clockwise rotation (feet point down)
    """
    if not keypoints or len(keypoints) < 3:
        return keypoints
    
    # ========== Parse keypoints ==========
    points = []
    for i in range(0, len(keypoints), 3):
        if i + 2 < len(keypoints):
            x, y, conf = keypoints[i], keypoints[i+1], keypoints[i+2]
            if conf > 0.1:  # Keep even low confidence for bounds
                points.append([x, y, conf])
    
    if len(points) < 2:
        return keypoints
    
    # ========== Find bounding box ==========
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    width = max_x - min_x
    height = max_y - min_y
    
    if width < 10 or height < 10:
        return keypoints  # Skip if too small
    
    # ========== Calculate scale with padding ==========
    padding_ratio = 0.15  # 15% padding around skeleton
    padding_x = width * padding_ratio
    padding_y = height * padding_ratio
    
    width_padded = width + 2 * padding_x
    height_padded = height + 2 * padding_y
    
    scale_x = canvas_w / width_padded if width_padded > 0 else 1.0
    scale_y = canvas_h / height_padded if height_padded > 0 else 1.0
    scale = min(scale_x, scale_y, 1.0)  # Don't enlarge
    
    # ========== Transform all keypoints ==========
    normalized = []
    for i in range(0, len(keypoints), 3):
        if i + 2 < len(keypoints):
            x, y, conf = keypoints[i], keypoints[i+1], keypoints[i+2]
            
            # Step 1: Center around bounding box
            x_centered = x - min_x
            y_centered = y - min_y
            
            # Step 2: Apply scale
            x_scaled = x_centered * scale
            y_scaled = y_centered * scale
            
            # Step 3: Add padding and center in canvas
            x_offset = padding_x * scale
            y_offset = padding_y * scale
            x_final = x_offset + (canvas_w - width_padded * scale) / 2 + x_scaled
            y_final = y_offset + (canvas_h - height_padded * scale) / 2 + y_scaled
            
            # Step 4: Apply 90° clockwise rotation so feet point downward
            # Formula: (x', y') = (height - y, x) around canvas center
            cx, cy = canvas_w / 2.0, canvas_h / 2.0
            x_centered_rot = x_final - cx
            y_centered_rot = y_final - cy
            x_rot = -y_centered_rot + cx
            y_rot = x_centered_rot + cy
            
            # Step 5: Ensure bounds
            x_rot = np.clip(x_rot, 0, canvas_w)
            y_rot = np.clip(y_rot, 0, canvas_h)
            
            normalized.extend([x_rot, y_rot, conf])
        else:
            normalized.extend([0, 0, 0])
    
    return normalized


def draw_skeleton(img, keypoints, color=(0, 255, 0), confidence_threshold=0.2,
                 thickness=2, kpt_size=3):
    """Draw skeleton on image."""
    if not keypoints or len(keypoints) < 3:
        return
    
    # Normalize skeleton to fit properly in frame
    keypoints = normalize_skeleton_to_frame(keypoints)
    
    # COCO pose keypoint connections
    LIMBS = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16)
    ]
    
    # Parse keypoints (x, y, conf) tuples
    points = []
    for i in range(0, len(keypoints), 3):
        if i + 2 < len(keypoints):
            x, y, conf = keypoints[i], keypoints[i+1], keypoints[i+2]
            if conf > confidence_threshold:
                x_int = int(np.clip(x, 0, img.shape[1] - 1))
                y_int = int(np.clip(y, 0, img.shape[0] - 1))
                points.append((x_int, y_int, conf))
            else:
                points.append(None)
        else:
            points.append(None)
    
    # Draw limbs
    for limb in LIMBS:
        pt1, pt2 = limb
        if pt1 < len(points) and pt2 < len(points):
            if points[pt1] is not None and points[pt2] is not None:
                p1 = (points[pt1][0], points[pt1][1])
                p2 = (points[pt2][0], points[pt2][1])
                cv2.line(img, p1, p2, color, thickness)
    
    # Draw keypoints
    for pt in points:
        if pt is not None:
            cv2.circle(img, (pt[0], pt[1]), kpt_size, color, -1)


class EnhancedVideoClipGenerator:
    """Enhanced video generator with better data loading and interpolation."""
    
    def __init__(self, manifest_path, data_dir, output_dir='Output',
                 fps=DEFAULT_FPS, interpolate=True, interpolation_frames=2):
        """
        Initialize generator.
        
        Args:
            manifest_path: Path to manifest JSON
            data_dir: Root data directory
            output_dir: Where to save videos
            fps: Video FPS
            interpolate: Whether to interpolate between frames
            interpolation_frames: Frames to generate between keyframes
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / 'individual_clips').mkdir(exist_ok=True)
        
        self.fps = fps
        self.frame_delay = 1  # For proper timing with interpolation
        self.interpolate = interpolate
        self.interpolation_frames = interpolation_frames if interpolate else 1
        
        #Load manifest
        logger.info(f"📂 Loading manifest: {manifest_path}")
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        
        # Split by label
        self.normal_people = [e for e in self.manifest if e.get('label') == 'normal']
        self.suspicious_people = [e for e in self.manifest if e.get('label') == 'suspicious']
        
        logger.info(f"   ✅ Normal sequences: {len(self.normal_people)}")
        logger.info(f"   ✅ Suspicious sequences: {len(self.suspicious_people)}")
    
    def load_person_frames(self, manifest_entry):
        """
        Load frames for a person from nested JSON structure.
        
        Args:
            manifest_entry: Dict with 'source_file', 'original_person_id', 'frame_range'
        
        Returns:
            List of dicts with 'keypoints' and metadata
        """
        source_file = manifest_entry['source_file']
        person_id = manifest_entry['original_person_id']
        frame_range = manifest_entry.get('frame_range', None)
        
        # Find the actual file
        for dataset_dir in [DATA_DIR / 'RetailS_train' / 'pose' / 'train',
                           DATA_DIR / 'RetailS_test_staged' / 'pose' / 'test',
                           DATA_DIR / 'RetailS_test_realworld' / 'pose' / 'test']:
            file_path = dataset_dir / source_file
            if file_path.exists():
                try:
                    with open(file_path) as f:
                        data = json.load(f)
                    
                    # Get frames for this person
                    if person_id in data:
                        frames_dict = data[person_id]
                        
                        # Load frames in order
                        frames_data = []
                        frame_indices = sorted([int(k) for k in frames_dict.keys()])
                        
                        for frame_idx in frame_indices:
                            frame_data = frames_dict[str(frame_idx)]
                            if 'keypoints' in frame_data:
                                frames_data.append({
                                    'keypoints': frame_data['keypoints'],
                                    'frame_id': frame_idx,
                                    'confidence': frame_data.get('scores', 0.5)
                                })
                        
                        return frames_data if frames_data else None
                
                except Exception as e:
                    logger.warning(f"Error loading {file_path}: {e}")
                    continue
        
        logger.warning(f"Could not find file for {source_file} (person {person_id})")
        return None
    
    def render_frame(self, keypoints, label, label_text):
        """Render a single frame with skeleton."""
        img = np.ones((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8) * COLOR_BACKGROUND
        
        color = COLOR_NORMAL if label == 'normal' else COLOR_SUSPICIOUS
        
        try:
            draw_skeleton(img, keypoints, color=color,
                         confidence_threshold=0.2, thickness=3, kpt_size=5)
            
            # Add label
            cv2.putText(img, label_text, (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            
            return img
        except Exception as e:
            logger.warning(f"Error rendering frame: {e}")
            return None
    
    def interpolate_sequence(self, frames_data):
        """
        Interpolate frames to make smoother sequences.
        
        Args:
            frames_data: List of frame dicts with 'keypoints'
        
        Returns:
            Expanded list with interpolated frames
        """
        if not self.interpolate or len(frames_data) < 2:
            return frames_data
        
        interpolated = []
        for i in range(len(frames_data) - 1):
            interpolated.append(frames_data[i])
            
            # Interpolate between this and next frame
            kp1 = frames_data[i]['keypoints']
            kp2 = frames_data[i+1]['keypoints']
            
            try:
                interp_kps = interpolate_keypoints(kp1, kp2, 
                                                   self.interpolation_frames + 1)
                
                # Add interpolated frames (skip first which is kp1)
                for interp_kp in interp_kps[1:]:
                    interpolated.append({
                        'keypoints': interp_kp,
                        'frame_id': f"{frames_data[i]['frame_id']}_interp",
                        'is_interpolated': True
                    })
            except Exception as e:
                logger.debug(f"Interpolation failed: {e}")
        
        # Add last frame
        interpolated.append(frames_data[-1])
        return interpolated
    
    def generate_individual_clips(self, people_list, label, num_clips):
        """Generate individual video clips."""
        print(f"\n🎬 Generating individual {label} clips...")
        
        clip_data = []
        
        for i, person_entry in enumerate(tqdm(people_list[:num_clips], 
                                              desc=f"  {label} clips")):
            # Load frames
            frames_data = self.load_person_frames(person_entry)
            if not frames_data or len(frames_data) < 2:
                continue
            
            # Interpolate for smoothness
            frames_data = self.interpolate_sequence(frames_data)
            
            # Render frames
            rendered_frames = []
            for frame_data in frames_data:
                img = self.render_frame(frame_data['keypoints'], label,
                                       f"{label.upper()} #{i+1}")
                if img is not None:
                    rendered_frames.append(img)
            
            if not rendered_frames:
                continue
            
            # Save individual clip
            clip_num = i + 1
            output_file = self.output_dir / 'individual_clips' / \
                         f'{label}_clip_{clip_num:03d}.mp4'
            self._write_video(output_file, rendered_frames)
            
            clip_data.append({
                'index': clip_num,
                'frames': frames_data,
                'rendered_frames': rendered_frames
            })
        
        return clip_data
    
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
            logger.info(f"   ✅ Saved: {output_path.name}")
            return True
        except Exception as e:
            logger.error(f"   ❌ Error writing {output_path.name}: {e}")
            return False
    
    def generate_grid_video(self, sequences, label, output_name, grid_cols=2):
        """Generate grid video from frame sequences."""
        print(f"\n📊 Generating {label} grid video...")
        
        if not sequences:
            print(f"   ❌ No sequences to grid")
            return False
        
        # Find max frames
        max_frames = max(len(seq['frames']) for seq in sequences)
        grid_rows = (len(sequences) + grid_cols - 1) // grid_cols
        
        grid_height = CANVAS_HEIGHT * grid_rows
        grid_width = CANVAS_WIDTH * grid_cols
        canvas_frames = []
        
        # For each time step
        for t in range(max_frames):
            canvas = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * COLOR_BACKGROUND
            
            for idx, sequence in enumerate(sequences):
                row = idx // grid_cols
                col = idx % grid_cols
                
                y_start = row * CANVAS_HEIGHT
                x_start = col * CANVAS_WIDTH
                y_end = y_start + CANVAS_HEIGHT
                x_end = x_start + CANVAS_WIDTH
                
                # Get frame at time t
                frame_idx = min(t, len(sequence['frames']) - 1)
                frame_data = sequence['frames'][frame_idx]
                
                # Render skeleton
                frame_img = self.render_frame(frame_data['keypoints'], label,
                                             f"{label.upper()}")
                if frame_img is not None:
                    canvas[y_start:y_end, x_start:x_end] = frame_img
                
                cv2.putText(canvas, f"Clip {idx+1}",
                           (x_start + 20, y_start + CANVAS_HEIGHT - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            
            # Repeat frame for timing
            for _ in range(self.frame_delay):
                canvas_frames.append(canvas)
        
        if canvas_frames:
            output_file = self.output_dir / output_name
            return self._write_video(output_file, canvas_frames)
        
        return False
    
    def generate_comparison_video(self, normal_sequences, suspicious_sequences, output_name):
        """Generate side-by-side comparison video."""
        print(f"\n🔄 Generating comparison video...")
        
        if not normal_sequences or not suspicious_sequences:
            print(f"   ❌ Need both normal and suspicious clips")
            return False
        
        min_clips = min(len(normal_sequences), len(suspicious_sequences))
        comparison_frames = []
        
        for pair_idx in range(min_clips):
            normal_seq = normal_sequences[pair_idx]
            suspicious_seq = suspicious_sequences[pair_idx]
            
            max_f = max(len(normal_seq['frames']), len(suspicious_seq['frames']))
            
            for t in range(max_f):
                canvas = np.ones((CANVAS_HEIGHT, CANVAS_WIDTH * 2, 3), dtype=np.uint8) * COLOR_BACKGROUND
                
                # Left: normal
                frame_idx = min(t, len(normal_seq['frames']) - 1)
                frame_img = self.render_frame(normal_seq['frames'][frame_idx]['keypoints'],
                                             'normal', 'NORMAL')
                if frame_img is not None:
                    canvas[:, :CANVAS_WIDTH] = frame_img
                
                # Right: suspicious
                frame_idx = min(t, len(suspicious_seq['frames']) - 1)
                frame_img = self.render_frame(suspicious_seq['frames'][frame_idx]['keypoints'],
                                             'suspicious', 'SUSPICIOUS')
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
    
    def generate_all(self, num_normal, num_suspicious, fps=None):
        """Generate all videos."""
        if fps:
            self.fps = fps
        
        mode = "INTERPOLATED" if self.interpolate else "STANDARD"
        print("\n" + "="*70)
        print(f"🎬 ENHANCED VIDEO CLIP GENERATION ({mode})")
        print("="*70)
        print(f"FPS: {self.fps}")
        print(f"Canvas: {CANVAS_WIDTH}x{CANVAS_HEIGHT}")
        print(f"Interpolation: {'ON' if self.interpolate else 'OFF'}")
        if self.interpolate:
            print(f"  Frames between keyframes: {self.interpolation_frames}")
        print(f"\n✨ Skeleton Processing:")
        print(f"  Normalization (centered, scale-invariant): ON ✓")
        print(f"  90° Rotation (feet pointing downward): ON ✓")
        print(f"  Padding ratio: 15%")
        print(f"\nClips to generate:")
        print(f"  Normal: {num_normal}")
        print(f"  Suspicious: {num_suspicious}")
        print("="*70)
        
        # Select random people
        selected_normal = random.sample(self.normal_people,
                                       min(num_normal, len(self.normal_people)))
        selected_suspicious = random.sample(self.suspicious_people,
                                           min(num_suspicious, len(self.suspicious_people)))
        
        # Generate individual clips
        normal_clips = self.generate_individual_clips(selected_normal, 'normal', num_normal)
        suspicious_clips = self.generate_individual_clips(selected_suspicious, 'suspicious', num_suspicious)
        
        # Generate grids
        if normal_clips:
            self.generate_grid_video(normal_clips, 'normal', 'normal_clips_grid.mp4', grid_cols=2)
        
        if suspicious_clips:
            self.generate_grid_video(suspicious_clips, 'suspicious', 'suspicious_clips_grid.mp4', grid_cols=2)
        
        # Generate comparison
        if normal_clips and suspicious_clips:
            self.generate_comparison_video(normal_clips, suspicious_clips,
                                          'normal_vs_suspicious.mp4')
        
        print("\n" + "="*70)
        print("✅ COMPLETE!")
        print("="*70)
        print(f"Output files in: {self.output_dir}/")
        print(f"  - individual_clips/ (all clips separately)")
        print(f"  - normal_clips_grid.mp4 (normal clips in grid)")
        print(f"  - suspicious_clips_grid.mp4 (suspicious clips in grid)")
        print(f"  - normal_vs_suspicious.mp4 (side-by-side comparison)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate enhanced video clips with interpolation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python generate_video_clips_v2.py --normal 5 --suspicious 5
  python generate_video_clips_v2.py --normal 10 --suspicious 10 --fps 20 --interpolate
  python generate_video_clips_v2.py --normal 3 --suspicious 3 --fps 30 --no-interpolate
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
    parser.add_argument('--manifest', type=str, default='manifest_all.json',
                       help='Manifest file to use')
    parser.add_argument('--interpolate', action='store_true', default=True,
                       help='Enable frame interpolation (default: enabled)')
    parser.add_argument('--no-interpolate', dest='interpolate', action='store_false',
                       help='Disable frame interpolation')
    parser.add_argument('--interp-frames', type=int, default=2,
                       help='Frames to generate between keyframes (default: 2)')
    
    args = parser.parse_args()
    
    # Find manifest
    manifest_path = MANIFESTS_DIR / args.manifest
    if not manifest_path.exists():
        # Try without Manifests dir
        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            print(f"❌ Error: Manifest not found: {args.manifest}")
            print(f"   Checked: {MANIFESTS_DIR}")
            return False
    
    # Generate videos
    try:
        generator = EnhancedVideoClipGenerator(
            manifest_path,
            DATA_DIR,
            args.output,
            fps=args.fps,
            interpolate=args.interpolate,
            interpolation_frames=args.interp_frames
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
