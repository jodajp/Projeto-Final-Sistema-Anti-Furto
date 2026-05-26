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
# Add root folder to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from Detecao.detector_factory import create_detector
from pipeline.config import AppConfig
from pipeline.video_source import create_video_source
from pipeline import SkeletonVisualizer, SpatialNormalizer

DISPLAY_SCALE_HEIGHT = 700  # Max height for display window (pixels)


# Visualization colors (BGR format for OpenCV)
COLOR_SKELETON_LINE = (0, 255, 255)   # Cyan
COLOR_SKELETON_NODE = (0, 0, 255)     # Red
COLOR_TEXT_HINT = (0, 255, 0)         # Green

def select_primary_person(keypoints: np.ndarray, scores: np.ndarray):
    """Return the highest-confidence person pose from detector output."""
    if keypoints is None or scores is None:
        return None

    kpts_arr = np.asarray(keypoints, dtype=np.float32)
    scores_arr = np.asarray(scores, dtype=np.float32)

    if kpts_arr.size == 0 or scores_arr.size == 0:
        return None

    if kpts_arr.ndim == 2:
        if kpts_arr.shape != (17, 2) or scores_arr.shape != (17,):
            return None
        return kpts_arr, scores_arr

    if kpts_arr.ndim != 3 or kpts_arr.shape[1:] != (17, 2):
        return None

    if scores_arr.ndim == 1 and scores_arr.shape == (17,):
        scores_arr = np.repeat(scores_arr[np.newaxis, :], kpts_arr.shape[0], axis=0)

    if scores_arr.ndim != 2 or scores_arr.shape[1] != 17:
        return None

    best_idx = int(np.argmax(scores_arr.mean(axis=1)))
    return kpts_arr[best_idx], scores_arr[best_idx]


def render_normalized_canvas(
    visualizer: SkeletonVisualizer,
    keypoints: np.ndarray,
    scores: np.ndarray,
    rotation_mode: int = 0,
) -> np.ndarray:
    """Render the normalized pose and optionally rotate the view."""
    canvas = visualizer.render(keypoints, scores, title="")

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
    app_config = AppConfig.from_file(str(config_path))

    # Process every frame for high temporal resolution
    app_config.data["runtime"]["frame_skip"] = 1

    detector = create_detector(app_config.detector_config())
    normalizer = SpatialNormalizer(app_config)
    skeleton_viz = SkeletonVisualizer(
        canvas_size=700,
        show_labels=False,
        show_confidence=False,
    )

    # Default rotation mode (180° keeps the normalized canvas oriented for review)
    rotation_mode = 0
    print("✓ Detector and pipeline normalizer ready")
    
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
        video_source = create_video_source({"id": str(vid_path)})
        cap = video_source.open()
        
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

            detected = detector.detect(frame)
            pose = select_primary_person(*detected) if detected else None

            if pose is None:
                norm_keypoints_list.append(np.full((17, 2), np.nan, dtype=np.float32))
                scores_list.append(np.zeros(17, dtype=np.float32))
            else:
                # Normalize pose using the shared pipeline normalizer -> pose[0] = keypoints, pose[1] = scores
                normalized_pose = normalizer.normalize(pose[0], pose[1])
                if normalized_pose.is_valid:
                    norm_keypoints_list.append(normalized_pose.keypoints.astype(np.float32))
                    scores_list.append(normalized_pose.scores.astype(np.float32))
                else:
                    norm_keypoints_list.append(np.full((17, 2), np.nan, dtype=np.float32))
                    scores_list.append(np.zeros(17, dtype=np.float32))
                
        cap.release()
        
        if not orig_frames:
            print(f"    ✗ Could not read frames")
            continue
        
        
        # ====================================================================
        # Interactive review UI
        # ====================================================================
        print(f"  Playing video (press Y/N/Q or R to rotate)...")
        
        canvas_h = skeleton_viz.canvas_size
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

            skel_canvas = render_normalized_canvas(
                skeleton_viz,
                norm_kpt,
                score,
                rotation_mode=rotation_mode_local,
            )

            if skel_canvas.shape[0] != canvas_h:
                skel_canvas = cv2.resize(skel_canvas, (canvas_h, canvas_h))

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
            kpt_array = np.nan_to_num(np.array(norm_keypoints_list), nan=0.0)   # (T, 17, 2)
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
