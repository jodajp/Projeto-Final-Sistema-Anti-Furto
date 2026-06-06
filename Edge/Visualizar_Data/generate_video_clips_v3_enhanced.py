#!/usr/bin/env python3
"""
Enhanced Video Clip Generator v3 - With Advanced Features
Incorporates confidence scoring, hand detection, motion analysis, and temporal information

Features:
1. Confidence-based joint filtering (ignore low-confidence poses)
2. Hand-position anomaly detection (hands near pockets/face)
3. Motion vector visualization
4. Temporal ground truth integration
5. Enhanced skeleton rendering with confidence indicators
"""

import cv2
import json
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
from tqdm import tqdm
import argparse
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    # Dataset paths
    DATA_DIR: Path = Path("Data")
    MANIFEST_DIR: Path = Path("Manifests")
    OUTPUT_DIR: Path = Path("output_v3")
    
    # Video settings
    FRAME_WIDTH: int = 800
    FRAME_HEIGHT: int = 1400
    FPS: int = 24
    
    # Skeleton settings
    CONFIDENCE_THRESHOLD: float = 0.5  # Filter joints below this confidence
    SHOW_CONFIDENCE_OVERLAY: bool = True
    
    # Hand detection - suspicious pocket/face access
    HAND_INDICES: List[int] = None  # COCO indices for hands
    POCKET_REGION: Tuple[int, int, int, int] = (200, 1000, 600, 1200)  # x1, y1, x2, y2
    FACE_REGION: Tuple[int, int, int, int] = (300, 900, 700, 1100)  # x1, y1, x2, y2
    
    # Motion settings
    SHOW_MOTION_VECTORS: bool = True
    MOTION_SCALE: float = 1.0
    
    def __post_init__(self):
        if self.HAND_INDICES is None:
            # COCO pose joints: 0-head, 5-7=left_hand, 9-11=right_hand
            self.HAND_INDICES = [9, 10, 11, 6, 7, 8]  # left and right hands
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = Config()

# ============================================================================
# COCO SKELETON DEFINITION
# ============================================================================

from Detecao.skeleton import SKELETON_CONNECTIONS, KEYPOINT_NAMES as COCO_JOINT_NAMES


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def unflatten_keypoints(flat_kp: List[float]) -> np.ndarray:
    """Convert flat 51-element list to 17×3 array (x,y,confidence)"""
    kp_array = np.array(flat_kp).reshape((17, 3))
    return kp_array

def get_joint_confidence(kp_array: np.ndarray, joint_idx: int) -> float:
    """Get confidence score for a joint"""
    return kp_array[joint_idx, 2]

def is_hand_in_pocket(hand_pos: Tuple[float, float], 
                      chest_pos: Tuple[float, float]) -> float:
    """Calculate hand-pocket proximity risk score (0-1)
    Higher = closer to pocket area
    """
    if hand_pos is None or chest_pos is None:
        return 0.0
    
    hand_x, hand_y = hand_pos
    chest_x, chest_y = chest_pos
    
    # Distance from hand to chest
    dist = np.sqrt((hand_x - chest_x)**2 + (hand_y - chest_y)**2)
    
    # Risk score: closer hands = higher risk
    # Max risk at 100 pixels (hand touching chest area)
    risk = max(0, 1.0 - (dist / 200.0))
    return risk

def calculate_motion_vector(current_pos: Optional[Tuple[float, float]],
                          previous_pos: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    """Calculate motion vector between frames"""
    if current_pos is None or previous_pos is None:
        return None
    return (current_pos[0] - previous_pos[0], current_pos[1] - previous_pos[1])

def interpolate_keypoints(keypoints_list: List) -> List[any]:
    """Interpolate keypoints between frames for smoother animation"""
    if len(keypoints_list) < 2:
        return keypoints_list
    
    # Create interpolation function for each coordinate
    frame_indices = np.arange(len(keypoints_list))
    
    # For each joint and coordinate (x, y)
    all_kps = [unflatten_keypoints(kp) for kp in keypoints_list]
    
    # Interpolate
    interp_kps = []
    for frame_idx in frame_indices:
        interp_kps.append(keypoints_list[int(frame_idx)])
        
        # Add interpolated frames between keypoints
        if frame_idx < len(keypoints_list) - 1:
            current_kp = unflatten_keypoints(keypoints_list[int(frame_idx)])
            next_kp = unflatten_keypoints(keypoints_list[int(frame_idx + 1)])
            
            # Interpolate 2 frames between each keyframe
            for t in [0.33, 0.66]:
                interp_kp = current_kp * (1 - t) + next_kp * t
                interp_kps.append(interp_kp.flatten().tolist())
    
    return interp_kps

def load_gt_labels(gt_file: Path) -> Optional[np.ndarray]:
    """Load ground truth labels for a video"""
    if gt_file.exists():
        return np.load(gt_file)
    return None

def normalize_skeleton_to_frame(kp_array: np.ndarray, 
                               confidence_threshold: float = 0.5) -> np.ndarray:
    """Normalize skeleton to fit in frame with intelligent scaling"""
    # Filter by confidence
    valid_joints = kp_array[:, 2] >= confidence_threshold
    valid_kps = kp_array[valid_joints]
    
    if len(valid_kps) == 0:
        return kp_array  # Return original if no valid joints
    
    x_coords = valid_kps[:, 0]
    y_coords = valid_kps[:, 1]
    
    if len(x_coords) == 0:
        return kp_array
    
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    # Add padding
    padding = 50
    width = x_max - x_min + 2 * padding
    height = y_max - y_min + 2 * padding
    
    # Calculate scaling
    scale = min(CONFIG.FRAME_WIDTH / width, CONFIG.FRAME_HEIGHT / height)
    scale = min(scale, 1.5)  # Don't scale up too much
    
    # Center in frame
    new_x_min = (CONFIG.FRAME_WIDTH - width * scale) / 2
    new_y_min = (CONFIG.FRAME_HEIGHT - height * scale) / 2
    
    # Apply transformation to all joints (keep low-confidence too)
    kp_normalized = kp_array.copy()
    kp_normalized[:, 0] = new_x_min + padding * scale + (kp_array[:, 0] - x_min) * scale
    kp_normalized[:, 1] = new_y_min + padding * scale + (kp_array[:, 1] - y_min) * scale
    
    # Clip to frame bounds
    kp_normalized[:, 0] = np.clip(kp_normalized[:, 0], 0, CONFIG.FRAME_WIDTH)
    kp_normalized[:, 1] = np.clip(kp_normalized[:, 1], 0, CONFIG.FRAME_HEIGHT)
    
    return kp_normalized

def draw_skeleton(frame: np.ndarray, 
                 kp_array: np.ndarray,
                 gt_label: Optional[int] = None,
                 hand_risk: float = 0.0,
                 motion_vectors: Optional[Dict[int, Tuple[float, float]]] = None) -> np.ndarray:
    """
    Draw skeleton with enhanced features:
    - Confidence-based coloring
    - Hand anomaly detection
    - Motion vectors
    - Ground truth label
    """
    frame_copy = frame.copy()
    
    # Flip Y-coordinate (dataset uses bottom-up, OpenCV uses top-down)
    kp_display = kp_array.copy()
    kp_display[:, 1] = CONFIG.FRAME_HEIGHT - kp_display[:, 1]
    
    # Draw skeleton edges
    for joint_from, joint_to in SKELETON_CONNECTIONS:
        if joint_from >= len(kp_display) or joint_to >= len(kp_display):
            continue
        
        pt1 = (int(kp_display[joint_from, 0]), int(kp_display[joint_from, 1]))
        pt2 = (int(kp_display[joint_to, 0]), int(kp_display[joint_to, 1]))
        
        conf_from = kp_display[joint_from, 2]
        conf_to = kp_display[joint_to, 2]
        
        if conf_from >= CONFIG.CONFIDENCE_THRESHOLD and conf_to >= CONFIG.CONFIDENCE_THRESHOLD:
            # Color based on avg confidence
            avg_conf = (conf_from + conf_to) / 2
            color = (0, int(255 * avg_conf), 255 - int(255 * avg_conf))  # Green=high conf, Red=low
            cv2.line(frame_copy, pt1, pt2, color, 2)
    
    # Draw joint circles
    for joint_idx, (x, y, conf) in enumerate(kp_display):
        if conf >= CONFIG.CONFIDENCE_THRESHOLD:
            pt = (int(x), int(y))
            
            # Color based on confidence
            if joint_idx in CONFIG.HAND_INDICES:
                # Hand joints - color based on pocket proximity risk
                if hand_risk > 0.5:
                    color = (0, 0, 255)  # Red - high risk
                else:
                    color = (0, 255, 0)  # Green - safe
            else:
                # Other joints - color by confidence
                color = (0, int(255 * conf), 255 - int(255 * conf))
            
            cv2.circle(frame_copy, pt, 4, color, -1)
            
            # Optional: draw confidence text
            if CONFIG.SHOW_CONFIDENCE_OVERLAY and conf < 0.9:
                cv2.putText(frame_copy, f"{conf:.1f}", 
                           (pt[0] + 5, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.3, (255, 255, 255), 1)
    
    # Draw motion vectors if provided
    if CONFIG.SHOW_MOTION_VECTORS and motion_vectors:
        for joint_idx, (vx, vy) in motion_vectors.items():
            if joint_idx >= len(kp_display):
                continue
            
            pt = (int(kp_display[joint_idx, 0]), int(kp_display[joint_idx, 1]))
            pt_end = (int(pt[0] + vx * CONFIG.MOTION_SCALE), 
                     int(pt[1] + vy * CONFIG.MOTION_SCALE))
            
            # Color: magnitude of motion
            mag = np.sqrt(vx**2 + vy**2)
            if mag > 5:
                cv2.arrowedLine(frame_copy, pt, pt_end, (255, 100, 0), 1, tipLength=0.3)
    
    # Draw ground truth label & risk score
    if gt_label is not None:
        y_pos = 30
        if gt_label == 1:
            cv2.putText(frame_copy, "SUSPICIOUS", (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:
            cv2.putText(frame_copy, "NORMAL", (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    if hand_risk > 0.3:
        cv2.putText(frame_copy, f"Hand Risk: {hand_risk:.1%}", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return frame_copy

# ============================================================================
# MAIN GENERATION FUNCTIONS
# ============================================================================

def generate_enhanced_clip(manifest_entry: Dict,
                         video_id: str,
                         output_path: Path) -> bool:
    """Generate enhanced video clip with all features"""
    
    # Determine dataset and load pose data
    dataset = manifest_entry.get('dataset', 'Training').replace(' ', '')
    source_file = manifest_entry['source_file']
    
    # Build pose file path based on dataset
    if 'Training' in manifest_entry.get('dataset', 'Training'):
        pose_file = CONFIG.DATA_DIR / 'RetailS_train' / 'pose' / 'train' / source_file
    elif 'Staged' in manifest_entry.get('dataset', 'Training'):
        pose_file = CONFIG.DATA_DIR / 'RetailS_test_staged' / 'pose' / 'test' / source_file
    else:  # Real-world
        pose_file = CONFIG.DATA_DIR / 'RetailS_test_realworld' / 'pose' / 'test' / source_file
    
    if not pose_file.exists():
        return False
    
    with open(pose_file) as f:
        pose_data = json.load(f)
    
    # Load ground truth if available (staged/realworld only)
    gt_labels = None
    if 'Test' in manifest_entry.get('dataset', 'Training'):
        if 'Staged' in manifest_entry.get('dataset', 'Training'):
            gt_file = CONFIG.DATA_DIR / 'RetailS_test_staged' / 'gt' / 'test_frame_mask' / source_file.replace('.json', '.npy')
        else:
            gt_file = CONFIG.DATA_DIR / 'RetailS_test_realworld' / 'gt' / 'test_frame_mask' / source_file.replace('.json', '.npy')
        gt_labels = load_gt_labels(gt_file)
    
    # Process frames
    camera_id = list(pose_data.keys())[0]
    person_id = manifest_entry.get('original_person_id', '0')
    
    # Get person data
    if person_id not in pose_data[camera_id]:
        return False
    
    person_data = pose_data[camera_id][person_id]
    person_keypoints = person_data.get('keypoints', [])
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, CONFIG.FPS, 
                         (CONFIG.FRAME_WIDTH, CONFIG.FRAME_HEIGHT))
    
    if not out.isOpened():
        print(f"  ⚠️ Failed to create video writer")
        return False
    
    # Process frames
    prev_kp = None
    for frame_idx, flat_kp in enumerate(person_keypoints):
        # Create blank frame
        frame = np.ones((CONFIG.FRAME_HEIGHT, CONFIG.FRAME_WIDTH, 3), dtype=np.uint8) * 240
        
        # Parse keypoints
        kp_array = unflatten_keypoints(flat_kp)
        
        # Normalize to frame
        kp_norm = normalize_skeleton_to_frame(kp_array, CONFIG.CONFIDENCE_THRESHOLD)
        
        # Calculate hand risk based on distance to torso
        hand_risk = 0.0
        if 9 < len(kp_norm) and 5 < len(kp_norm):
            hand_pos = (kp_norm[9, 0], kp_norm[9, 1])
            chest_pos = (kp_norm[5, 0], kp_norm[5, 1])
            hand_risk = is_hand_in_pocket(hand_pos, chest_pos)
        
        # Calculate motion vectors
        motion_vectors = None
        if CONFIG.SHOW_MOTION_VECTORS and prev_kp is not None:
            motion_vectors = {}
            prev_kp_array = unflatten_keypoints(prev_kp)
            for j in range(len(kp_norm)):
                motion = calculate_motion_vector(
                    (kp_norm[j, 0], kp_norm[j, 1]),
                    (prev_kp_array[j, 0], prev_kp_array[j, 1])
                )
                if motion:
                    motion_vectors[j] = motion
        
        # Get GT label if available
        gt_label = None
        if gt_labels is not None and frame_idx < len(gt_labels):
            gt_label = int(gt_labels[frame_idx])
        
        # Draw skeleton
        frame = draw_skeleton(frame, kp_norm, gt_label, hand_risk, motion_vectors)
        
        # Write frame
        out.write(frame)
        prev_kp = flat_kp
    
    out.release()
    return True

def main():
    parser = argparse.ArgumentParser(description="Enhanced Video Clip Generator v3")
    parser.add_argument('--manifest', default='manifest_all.json', help='Manifest file')
    parser.add_argument('--normal', type=int, default=1, help='Number of normal clips')
    parser.add_argument('--suspicious', type=int, default=1, help='Number of suspicious clips')
    parser.add_argument('--show-confidence', action='store_true', default=True)
    parser.add_argument('--show-motion', action='store_true', default=True)
    
    args = parser.parse_args()
    
    # Load manifest
    manifest_path = CONFIG.MANIFEST_DIR / args.manifest
    if not manifest_path.exists():
        print(f"❌ Manifest not found: {manifest_path}")
        return
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    print(f"📊 Manifest loaded: {len(manifest)} sequences")
    
    # Process clips
    normal_count = 0
    suspicious_count = 0
    
    for entry in tqdm(manifest, desc="Generating clips"):
        is_suspicious = entry.get('label', 'normal').lower() == 'suspicious'
        
        if is_suspicious and suspicious_count < args.suspicious:
            output_file = CONFIG.OUTPUT_DIR / f"suspicious_{suspicious_count:03d}.mp4"
            if generate_enhanced_clip(entry, f"susp_{suspicious_count}", output_file):
                suspicious_count += 1
        
        elif not is_suspicious and normal_count < args.normal:
            output_file = CONFIG.OUTPUT_DIR / f"normal_{normal_count:03d}.mp4"
            if generate_enhanced_clip(entry, f"norm_{normal_count}", output_file):
                normal_count += 1
        
        if normal_count >= args.normal and suspicious_count >= args.suspicious:
            break
    
    print(f"\n✅ Complete: {normal_count} normal, {suspicious_count} suspicious clips")

if __name__ == "__main__":
    main()
