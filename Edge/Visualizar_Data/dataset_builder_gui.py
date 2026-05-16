#!/usr/bin/env python3
"""
Custom Dataset Preparation Tool for Shoplifting Videos

Processes videos from the 'Shoplifting' folder, extracts normalized skeleton poses
for the main person (highest confidence), and provides an interactive UI for
accepting/rejecting each video. Generates a .pkl file compatible with PySkl/MMAction2.

Features:
  - Automatic single-person extraction (ignores background people)
  - Full skeleton normalization with centering and scaling
  - Side-by-side visualization: original video + normalized skeleton
  - Real-time rotation correction (R key to cycle through 4 rotation modes)

Usage:
  python Visualizar_Data/dataset_builder_gui.py

Controls:
  Y - Accept video (save to dataset as 'suspicious' label)
  N - Reject video (discard)
  Q - Quit early (save progress and exit)
  R - Rotate skeleton (cycle through 4 orientations)
"""
import sys
import cv2
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml

# Add root folder to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from Detecao.detector_factory import create_detector
from pipeline.config import AppConfig
from Visualizar_Data.skeleton_normalizer import SkeletonNormalizer, SkeletonNormConfig


# COCO 17 keypoint skeleton connections (limbs to draw)
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2),           # nose -> eyes
    (1, 3), (2, 4),           # eyes -> ears
    (5, 6),                   # shoulders
    (5, 7), (6, 8),           # shoulders -> elbows
    (7, 9), (8, 10),          # elbows -> wrists
    (11, 12),                 # hips (pelvis)
    (5, 11), (6, 12),         # shoulders -> hips
    (11, 13), (12, 14),       # hips -> knees
    (13, 15), (14, 16),       # knees -> ankles
]

# Visualization colors (BGR format for OpenCV)
COLOR_SKELETON_LINE = (0, 255, 255)   # Cyan
COLOR_SKELETON_NODE = (0, 0, 255)     # Red
COLOR_TEXT_HINT = (0, 255, 0)         # Green

# Canvas configuration
DISPLAY_SCALE_HEIGHT = 700  # Max height for display window (pixels)
MIN_CONFIDENCE = 0.3        # Minimum confidence to draw keypoint

def draw_skeleton_on_canvas(keypoints: np.ndarray, scores: np.ndarray, 
                           canvas_size: tuple = (1400, 800), 
                           conf_threshold: float = 0.3, 
                           rotation_mode: int = 0) -> np.ndarray:
    """
    Render normalized skeleton keypoints onto a blank canvas.
    
    Args:
        keypoints: Shape (17, 2), pixel coordinates from normalizer
        scores: Shape (17,), confidence scores [0, 1]
        canvas_size: (height, width) tuple matching normalizer canvas
        conf_threshold: Minimum confidence to draw keypoint/limb
        rotation_mode: 0=none, 1=90° CW, 2=180°, 3=90° CCW
    
    Returns:
        Canvas image, shape (H, W, 3), BGR uint8
    """
    height, width = canvas_size
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    
    if keypoints is None or len(keypoints) == 0:
        return canvas

    # Draw skeleton limbs
    for idx_from, idx_to in SKELETON_CONNECTIONS:
        pt_from = keypoints[idx_from]
        pt_to = keypoints[idx_to]
        
        if np.isnan(pt_from).any() or np.isnan(pt_to).any():
            continue
            
        conf_avg = (scores[idx_from] + scores[idx_to]) / 2.0
        if conf_avg < conf_threshold:
            continue
            
        pt_from_int = tuple(map(int, pt_from))
        pt_to_int = tuple(map(int, pt_to))
        cv2.line(canvas, pt_from_int, pt_to_int, COLOR_SKELETON_LINE, thickness=2)
        
    # Draw skeleton keypoints (nodes)
    for i, pt in enumerate(keypoints):
        if np.isnan(pt).any() or scores[i] < conf_threshold:
            continue
        pt_int = tuple(map(int, pt))
        cv2.circle(canvas, pt_int, radius=4, color=COLOR_SKELETON_NODE, thickness=-1)
    
    # Apply rotation correction
    if rotation_mode == 1:
        canvas = cv2.rotate(canvas, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_mode == 2:
        canvas = cv2.rotate(canvas, cv2.ROTATE_180)
    elif rotation_mode == 3:
        canvas = cv2.rotate(canvas, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
    return canvas

def main():
    """Main entry point for dataset preparation."""
    print("=" * 70)
    print("CUSTOM DATASET PREPARATION TOOL (Shoplifting)")
    print("=" * 70)
    
    # ======================================================================
    # Initialize detector and normalizer
    # ======================================================================
    print("\n[1/4] Loading configuration and detector...")
    config_path = ROOT_DIR / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        yaml_conf = yaml.safe_load(f)
    
    # Process every frame for high temporal resolution
    yaml_conf['runtime']['frame_skip'] = 1 
    
    app_config = AppConfig(yaml_conf)
    detector = create_detector(app_config.detector_config())
    
    # Configure skeleton normalization
    norm_config = SkeletonNormConfig()
    norm_config.apply_rotation_90deg = False  # Natural upright orientation
    normalizer = SkeletonNormalizer(norm_config)
    
    # Default rotation mode (180° provides correct orientation + scale)
    rotation_mode = 2
    print("✓ Detector and normalizer ready")
    
    # ======================================================================
    # Discover videos
    # ======================================================================
    print("\n[2/4] Discovering videos...")
    data_dir = ROOT_DIR / "Visualizar_Data" / "Data" / "Shoplifting"
    out_dir = ROOT_DIR / "Visualizar_Data" / "Output"
    out_dir.mkdir(exist_ok=True)
    
    custom_dataset_file = out_dir / "custom_shoplifting_dataset.pkl"
    
    videos = sorted(list(data_dir.glob("*.mp4")) + list(data_dir.glob("*.avi")))
    if not videos:
        print(f"✗ No videos found in {data_dir}")
        print("  Please add .mp4 or .avi files to the Shoplifting folder.")
        return

    print(f"✓ Found {len(videos)} videos")
    
    # ======================================================================
    # Process each video
    # ======================================================================
    print("\n[3/4] Processing videos (extract poses and review)...")
    
    
    annotations = []
    video_names = []
    
    for video_idx, vid_path in enumerate(videos, start=1):
        print(f"\n  [{video_idx}/{len(videos)}] {vid_path.name}")
        cap = cv2.VideoCapture(str(vid_path))
        
        orig_frames = []
        norm_keypoints_list = []
        scores_list = []
        
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Extract poses from all frames
        for _ in tqdm(range(total_frames), desc="  Extracting poses", leave=False):
            ret, frame = cap.read()
            if not ret:
                break
                
            orig_frames.append(frame.copy())
            
            # Pose detection
            kpts, scores = detector.detect(frame)
            kpts = np.array(kpts)
            scores = np.array(scores)
            
            # Single-person extraction (take person with highest avg confidence)
            if kpts is not None and len(kpts) > 0:
                if len(kpts.shape) == 3:  # (num_people, 17, 2)
                    person_confidences = scores.mean(axis=1)
                    best_idx = np.argmax(person_confidences)
                    best_kpt = kpts[best_idx]
                    best_score = scores[best_idx]
                else:  # Already single person
                    best_kpt = kpts
                    best_score = scores
                
                # Normalize: combine to (17, 3) format [x, y, score]
                combined_kpt = np.column_stack((best_kpt, best_score))
                norm_kpt = normalizer.normalize_and_center(combined_kpt)
                
                # Extract normalized coordinates and scores
                norm_keypoints_list.append(norm_kpt[:, :2])   # (17, 2)
                scores_list.append(norm_kpt[:, 2])            # (17,)
            else:
                # No detection - use placeholder
                norm_keypoints_list.append(np.zeros((17, 2)))
                scores_list.append(np.zeros((17,)))
                
        cap.release()
        
        if not orig_frames:
            print(f"    ✗ Could not read frames")
            continue
        
        
        # ====================================================================
        # Interactive review UI
        # ====================================================================
        print(f"  Playing video (press Y/N/Q or R to rotate)...")
        
        canvas_h, canvas_w = norm_config.canvas_height, norm_config.canvas_width
        decision = None
        frame_idx = 0
        rotation_mode_local = rotation_mode
        
        while True:
            frame = orig_frames[frame_idx]
            norm_kpt = norm_keypoints_list[frame_idx]
            score = scores_list[frame_idx]
            
            # Prepare original frame (resize to match canvas height)
            scale = canvas_h / orig_h
            new_w = int(orig_w * scale)
            frame_resized = cv2.resize(frame, (new_w, canvas_h))
            
            # Draw normalized skeleton
            skel_canvas = draw_skeleton_on_canvas(
                norm_kpt, score, 
                canvas_size=(canvas_h, canvas_w),
                conf_threshold=MIN_CONFIDENCE,
                rotation_mode=rotation_mode_local
            )
            
            # Handle dimension swap from rotation
            skel_h, skel_w = skel_canvas.shape[:2]
            if skel_h != canvas_h:
                skel_canvas = cv2.resize(skel_canvas, (canvas_w, canvas_h))
            
            # Concatenate side-by-side
            side_by_side = np.hstack([frame_resized, skel_canvas])
            
            # Scale down for display (fit on monitor)
            display_scale = DISPLAY_SCALE_HEIGHT / canvas_h
            disp_w = int(side_by_side.shape[1] * display_scale)
            disp_h = int(side_by_side.shape[0] * display_scale)
            side_by_side_disp = cv2.resize(side_by_side, (disp_w, disp_h))
            
            # Add info text
            rotation_labels = ["No Rotation", "90° CW", "180°", "90° CCW"]
            hint_text = f"Y:Keep | N:Discard | Q:Quit | R:Rotate[{rotation_labels[rotation_mode_local]}] | Frame: {frame_idx+1}/{len(orig_frames)}"
            cv2.putText(side_by_side_disp, hint_text, 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT_HINT, 2)
            
            cv2.imshow("Dataset Preparation: Original (Left) | Normalized (Right)", side_by_side_disp)
            
            # Wait for key input (~30 FPS playback)
            key = cv2.waitKey(33) & 0xFF
            
            if key == ord('y') or key == ord('Y'):
                decision = 'keep'
                break
            elif key == ord('n') or key == ord('N'):
                decision = 'discard'
                break
            elif key == ord('q') or key == ord('Q'):
                decision = 'quit'
                break
            elif key == ord('r') or key == ord('R'):
                rotation_mode_local = (rotation_mode_local + 1) % 4
                continue
                
            # Auto-loop through frames
            frame_idx = (frame_idx + 1) % len(orig_frames)
        
        # ====================================================================
        # Process decision
        # ====================================================================
        if decision == 'keep':
            # Convert to PySkl format: (1, T, 17, 2) and (1, T, 17)
            kpt_array = np.array(norm_keypoints_list)   # (T, 17, 2)
            score_array = np.array(scores_list)         # (T, 17)
            
            annotations.append({
                'frame_dir': vid_path.stem,
                'label': 'suspicious',  # All Shoplifting videos labeled as suspicious
                'img_shape': (orig_h, orig_w),
                'original_shape': (orig_h, orig_w),
                'total_frames': len(kpt_array),
                'keypoint': np.expand_dims(kpt_array, axis=0),         # (1, T, 17, 2)
                'keypoint_score': np.expand_dims(score_array, axis=0)  # (1, T, 17)
            })
            video_names.append(vid_path.stem)
            print(f"    ✓ Saved to dataset ({len(kpt_array)} frames)")
            
        elif decision == 'discard':
            print(f"    - Discarded")
            
        elif decision == 'quit':
            print(f"\n  User requested early exit (saved progress)...")
            break
    
    cv2.destroyAllWindows()
    
    # ======================================================================
    # Save dataset pickle
    # ======================================================================
    print("\n[4/4] Saving dataset...")
    
    if annotations:
        dataset_dict = {
            'split': {'train': video_names},
            'annotations': annotations
        }
        with open(custom_dataset_file, 'wb') as f:
            pickle.dump(dataset_dict, f)
        
        print(f"✓ Dataset saved: {custom_dataset_file.name}")
        print(f"  - Total samples: {len(annotations)}")
        print(f"  - Total frames: {sum(a['total_frames'] for a in annotations)}")
        print(f"\nYou can now use this dataset with PySkl/MMAction2 for training!")
    else:
        print("✗ No videos selected. No dataset created.")
    
    print("\n" + "=" * 70)
        

if __name__ == '__main__':
    main()
