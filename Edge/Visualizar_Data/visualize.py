#!/usr/bin/env python3
"""
RetailS Dataset Interactive Visualizer - Enhanced Edition

Complete rewrite with manifest support, confidence scoring, hand detection

Features:
- Manifest-based data loading (correct JSON structure)
- Confidence-based joint filtering
- Hand-pocket proximity risk detection
- Motion velocity visualization
- Ground truth label integration
- Interactive zoom, pan, person selection
"""

import json
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# COCO 17-point skeleton
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (5, 11), (6, 12), (11, 12),  # torso
    (11, 13), (13, 15), (12, 14), (14, 16)  # legs
]

COCO_JOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

HAND_INDICES = [9, 10, 11, 6, 7, 8]  # Wrists and elbows
TORSO_INDICES = [5, 6, 11, 12]  # Shoulders and hips

# Canvas dimensions (portrait - retail overhead)
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 1400

# Confidence thresholds
CONF_HIGH = 0.7
CONF_MED = 0.5
CONF_LOW = 0.3

# ============================================================================
# COLORS
# ============================================================================

def confidence_color(conf):
    """Color by confidence level"""
    if conf >= CONF_HIGH:
        return (0, 255, 0)  # Green
    elif conf >= CONF_MED:
        return (0, 255, 255)  # Yellow
    else:
        return (0, 0, 255)  # Red

def get_person_color(person_id):
    """Unique color per person"""
    h = (int(person_id) * 137.5) % 180
    hsv = np.uint8([[[h, 255, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return tuple(int(x) for x in bgr)

# ============================================================================
# DATA LOADING
# ============================================================================

class RetailSDataLoader:
    """Load RetailS dataset from manifests"""
    
    def __init__(self, data_root="Data", manifest="Manifests/manifest_all.json"):
        self.data_root = Path(data_root)
        self.manifest = {}
        self.load_manifest(manifest)
        self.current_entry = None
        self.pose_data = None
        self.current_label = None
        
    def load_manifest(self, manifest_file):
        """Load manifest JSON"""
        mf = Path(manifest_file)
        if mf.exists():
            with open(mf) as f:
                entries = json.load(f)
                self.manifest = {e['person_id']: e for e in entries}
        print(f"✓ Loaded {len(self.manifest)} sequences")
    
    def load_sequence(self, person_id):
        """Load pose data for a person"""
        if person_id not in self.manifest:
            return False
        
        entry = self.manifest[person_id]
        self.current_entry = entry
        
        # Build pose file path
        dataset = entry.get('dataset', 'Training')
        source_file = entry['source_file']
        
        if 'Training' in dataset:
            pose_file = self.data_root / 'RetailS_train' / 'pose' / 'train' / source_file
        elif 'Staged' in dataset:
            pose_file = self.data_root / 'RetailS_test_staged' / 'pose' / 'test' / source_file
        else:
            pose_file = self.data_root / 'RetailS_test_realworld' / 'pose' / 'test' / source_file
        
        if not pose_file.exists():
            return False
        
        # Load pose data
        with open(pose_file) as f:
            full_data = json.load(f)
        
        # Extract person's data
        camera_id = entry.get('original_camera', '1')
        person_orig_id = entry.get('original_person_id', '0')
        
        if camera_id not in full_data or person_orig_id not in full_data[camera_id]:
            return False
        
        self.pose_data = full_data[camera_id][person_orig_id]
        
        # Load labels
        self.current_label = None
        if 'Test' in dataset:
            gt_file = pose_file.parent.parent / 'gt' / 'test_frame_mask' / source_file.replace('.json', '.npy')
            if gt_file.exists():
                gt = np.load(gt_file)
                person_idx = int(person_orig_id)
                if person_idx < len(gt):
                    self.current_label = int(gt[person_idx])
        
        return True
    
    def get_keypoints(self):
        """Get keypoint sequence (wrapped as single-frame list)"""
        if self.pose_data is None:
            return []
        # JSON contains single 51-element frame, wrap in list for frame indexing
        kp = self.pose_data.get('keypoints', [])
        return [kp] if len(kp) == 51 else []
    
    def get_sequences(self, filter_label=None):
        """Get sequence IDs, optionally filtered"""
        if filter_label is None:
            return list(self.manifest.keys())
        
        result = []
        for pid, entry in self.manifest.items():
            if entry.get('label', 'normal').lower() == filter_label.lower():
                result.append(pid)
        return result

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def unflatten_keypoints(flat_kp):
    """Convert 51-element list to 17x3 array"""
    return np.array(flat_kp, dtype=np.float32).reshape((17, 3))

def normalize_skeleton(kp_array, conf_threshold=CONF_MED):
    """Normalize skeleton to canvas"""
    valid = kp_array[kp_array[:, 2] >= conf_threshold]
    
    if len(valid) < 3:
        return kp_array
    
    x_min, x_max = valid[:, 0].min(), valid[:, 0].max()
    y_min, y_max = valid[:, 1].min(), valid[:, 1].max()
    
    width = max(x_max - x_min + 100, 1)
    height = max(y_max - y_min + 100, 1)
    
    scale = min(CANVAS_WIDTH / width, CANVAS_HEIGHT / height, 1.5)
    
    offset_x = (CANVAS_WIDTH - (x_max - x_min) * scale) / 2
    offset_y = (CANVAS_HEIGHT - (y_max - y_min) * scale) / 2
    
    kp_norm = kp_array.copy()
    kp_norm[:, 0] = offset_x + (kp_array[:, 0] - x_min) * scale
    kp_norm[:, 1] = offset_y + (kp_array[:, 1] - y_min) * scale
    kp_norm[:, 0] = np.clip(kp_norm[:, 0], 0, CANVAS_WIDTH)
    kp_norm[:, 1] = np.clip(kp_norm[:, 1], 0, CANVAS_HEIGHT)
    
    return kp_norm

def calculate_hand_risk(kp_array):
    """Calculate hand-pocket proximity risk (0-1)"""
    valid = kp_array[:, 2] >= CONF_MED
    hand_joints = [j for j in HAND_INDICES if j < len(valid) and valid[j]]
    torso_joints = [j for j in TORSO_INDICES if j < len(valid) and valid[j]]
    
    if not hand_joints or not torso_joints:
        return 0.0
    
    hand_pos = kp_array[hand_joints, :2].mean(axis=0)
    torso_pos = kp_array[torso_joints, :2].mean(axis=0)
    dist = np.linalg.norm(hand_pos - torso_pos)
    
    return float(max(0, 1.0 - (dist / 250.0)))

def calculate_motion(kp_current, kp_previous):
    """Calculate motion magnitude"""
    if kp_current is None or kp_previous is None:
        return None
    
    kp_c = unflatten_keypoints(kp_current)
    kp_p = unflatten_keypoints(kp_previous)
    
    valid = (kp_c[:, 2] >= CONF_MED) & (kp_p[:, 2] >= CONF_MED)
    if not valid.any():
        return None
    
    motion = kp_c[valid, :2] - kp_p[valid, :2]
    return float(np.linalg.norm(motion, axis=1).mean())

# ============================================================================
# RENDERING
# ============================================================================

def draw_skeleton(frame, kp_array, person_id=None, gt_label=None, hand_risk=0.0, 
                 motion=None, conf_threshold=CONF_MED):
    """Render skeleton with analyses"""
    
    kp_norm = normalize_skeleton(kp_array, conf_threshold)
    
    # Draw skeleton edges
    for j1, j2 in COCO_SKELETON:
        if kp_norm[j1, 2] < conf_threshold or kp_norm[j2, 2] < conf_threshold:
            continue
        
        pt1 = tuple(kp_norm[j1, :2].astype(int))
        pt2 = tuple(kp_norm[j2, :2].astype(int))
        conf_avg = (kp_norm[j1, 2] + kp_norm[j2, 2]) / 2
        color = confidence_color(conf_avg)
        cv2.line(frame, pt1, pt2, color, 2)
    
    # Draw joints
    for i, (x, y, conf) in enumerate(kp_norm):
        if conf < conf_threshold:
            continue
        
        pt = tuple([int(x), int(y)])
        color = confidence_color(conf)
        
        # Hand risk coloring
        if i in HAND_INDICES and hand_risk > 0.3:
            color = (0, 0, 255)
        
        cv2.circle(frame, pt, 4, color, -1)
        
        if conf < 0.8:
            cv2.putText(frame, f"{conf:.1f}", (pt[0]+5, pt[1]-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
    
    # GT label
    if gt_label is not None:
        color = (0, 0, 255) if gt_label == 1 else (0, 255, 0)
        text = "SUSPICIOUS" if gt_label == 1 else "NORMAL"
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    
    # Hand risk
    if hand_risk > 0.2:
        color = (0, int(255 * (1 - hand_risk)), int(255 * hand_risk))
        cv2.putText(frame, f"Hand Risk: {hand_risk:.0%}", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Motion
    if motion is not None and motion > 0:
        motion_text = f"Motion: {motion:.1f} px/frame"
        color = (0, 255, 0) if motion < 10 else (0, 165, 255) if motion < 20 else (0, 0, 255)
        cv2.putText(frame, motion_text, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# ============================================================================
# INTERACTIVE VIEWER
# ============================================================================

def visualize_interactive(loader, start_person=None):
    """Interactive matplotlib-based viewer"""
    
    all_sequences = loader.get_sequences()
    if not all_sequences:
        print("No sequences loaded")
        return
    
    current_idx = 0
    if start_person and start_person in all_sequences:
        current_idx = all_sequences.index(start_person)
    
    # Initial load
    current_person = all_sequences[current_idx]
    loader.load_sequence(current_person)
    keypoints = loader.get_keypoints()
    
    # Setup figure
    fig, ax = plt.subplots(figsize=(12, 18))
    plt.subplots_adjust(bottom=0.2)
    
    canvas = ax.imshow(np.ones((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8) * 240)
    ax.set_xlim(0, CANVAS_WIDTH)
    ax.set_ylim(CANVAS_HEIGHT, 0)
    
    # Frame slider
    ax_slider = plt.axes([0.2, 0.10, 0.6, 0.02])
    slider = Slider(ax_slider, 'Frame', 0, max(0, len(keypoints) - 1), valinit=0, valstep=1)
    
    # Buttons
    ax_prev = plt.axes([0.05, 0.03, 0.06, 0.03])
    ax_next = plt.axes([0.12, 0.03, 0.06, 0.03])
    ax_prev_seq = plt.axes([0.19, 0.03, 0.08, 0.03])
    ax_next_seq = plt.axes([0.28, 0.03, 0.08, 0.03])
    ax_filter_susp = plt.axes([0.37, 0.03, 0.08, 0.03])
    ax_filter_norm = plt.axes([0.46, 0.03, 0.08, 0.03])
    
    btn_prev = Button(ax_prev, '◀ Frame')
    btn_next = Button(ax_next, 'Frame ▶')
    btn_prev_seq = Button(ax_prev_seq, '◀◀ Seq')
    btn_next_seq = Button(ax_next_seq, 'Seq ▶▶')
    btn_filter_susp = Button(ax_filter_susp, 'Suspicious')
    btn_filter_norm = Button(ax_filter_norm, 'Normal')
    
    state = {'frame': 0, 'filter': 'all', 'keypoints': keypoints}
    
    def update_frame(frame_idx):
        """Redraw current frame"""
        frame = np.ones((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8) * 240
        
        kp_list = state['keypoints']
        if frame_idx < len(kp_list):
            kp = unflatten_keypoints(kp_list[frame_idx])
            
            motion = None
            if frame_idx > 0:
                motion = calculate_motion(kp_list[frame_idx], kp_list[frame_idx - 1])
            
            hand_risk = calculate_hand_risk(kp)
            draw_skeleton(frame, kp, gt_label=loader.current_label, 
                         hand_risk=hand_risk, motion=motion)
        
        # Info
        info = f"Seq: {current_person} Frame: {frame_idx + 1}/{len(kp_list)}"
        if loader.current_label is not None:
            info += f" | GT: {'SUSP' if loader.current_label else 'NORM'}"
        cv2.putText(frame, info, (20, CANVAS_HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (0, 0, 0), 1)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        canvas.set_data(frame_rgb)
        fig.canvas.draw_idle()
    
    def on_frame_prev(evt):
        state['frame'] = max(0, state['frame'] - 1)
        slider.set_val(state['frame'])
    
    def on_frame_next(evt):
        state['frame'] = min(len(state['keypoints']) - 1, state['frame'] + 1)
        slider.set_val(state['frame'])
    
    def on_seq_prev(evt):
        nonlocal current_idx, current_person
        current_idx = max(0, current_idx - 1)
        current_person = all_sequences[current_idx]
        loader.load_sequence(current_person)
        state['keypoints'] = loader.get_keypoints()
        state['frame'] = 0
        slider.set_val(0)
        slider.vmax = max(0, len(state['keypoints']) - 1)
        update_frame(0)
    
    def on_seq_next(evt):
        nonlocal current_idx, current_person
        current_idx = min(len(all_sequences) - 1, current_idx + 1)
        current_person = all_sequences[current_idx]
        loader.load_sequence(current_person)
        state['keypoints'] = loader.get_keypoints()
        state['frame'] = 0
        slider.set_val(0)
        slider.vmax = max(0, len(state['keypoints']) - 1)
        update_frame(0)
    
    def on_filter_susp(evt):
        nonlocal current_idx, current_person, all_sequences
        filtered = loader.get_sequences(filter_label='suspicious')
        if filtered:
            all_sequences = filtered
            current_idx = 0
            current_person = all_sequences[current_idx]
            loader.load_sequence(current_person)
            state['keypoints'] = loader.get_keypoints()
            state['frame'] = 0
            slider.set_val(0)
            slider.vmax = max(0, len(state['keypoints']) - 1)
            update_frame(0)
    
    def on_filter_norm(evt):
        nonlocal current_idx, current_person, all_sequences
        filtered = loader.get_sequences(filter_label='normal')
        if filtered:
            all_sequences = filtered
            current_idx = 0
            current_person = all_sequences[current_idx]
            loader.load_sequence(current_person)
            state['keypoints'] = loader.get_keypoints()
            state['frame'] = 0
            slider.set_val(0)
            slider.vmax = max(0, len(state['keypoints']) - 1)
            update_frame(0)
    
    def on_slider(val):
        state['frame'] = int(slider.val)
        update_frame(state['frame'])
    
    btn_prev.on_clicked(on_frame_prev)
    btn_next.on_clicked(on_frame_next)
    btn_prev_seq.on_clicked(on_seq_prev)
    btn_next_seq.on_clicked(on_seq_next)
    btn_filter_susp.on_clicked(on_filter_susp)
    btn_filter_norm.on_clicked(on_filter_norm)
    slider.on_changed(on_slider)
    
    update_frame(0)
    plt.show()

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    loader = RetailSDataLoader()
    visualize_interactive(loader)
