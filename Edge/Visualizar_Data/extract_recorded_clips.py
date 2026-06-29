import sys
import os
from pathlib import Path
import json
import numpy as np
import cv2
import yaml

ROOT_DIR = Path(r"f:\Github\Projeto-Final-Sistema-Anti-Furto")
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "Edge"))

from Edge.Detecao.detector_factory import create_detector
from Edge.pipeline.config import AppConfig
from Edge.pipeline.spatial_normalizer import SpatialNormalizer, NormalizationParams

from Edge.pipeline.temporal_pose_filter import TemporalPoseFilter

def process_video(video_path, detector, normalizer, output_npz_path, label, use_filter=False, frame_skip=1, app_config=None):
    print(f"Processing {video_path} (Label: {label}, Filter: {use_filter}, Skip: {frame_skip})...")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return False
        
    kpts_list = []
    scores_list = []
    norm_kpts_list = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    
    temp_filter = None
    if use_filter and app_config:
        temp_filter = TemporalPoseFilter(app_config)
    
    while True:
        if frame_skip > 1:
            for _ in range(frame_skip - 1):
                cap.grab()
                frame_idx += 1
                
        ret, frame = cap.read()
        if not ret:
            break
            
        kpts, scores = detector.detect(frame)
        if kpts is None or len(kpts) == 0:
            raw_kpts = np.zeros((17, 2), dtype=np.float32)
            raw_scores = np.zeros(17, dtype=np.float32)
        else:
            kpts = np.asarray(kpts, dtype=np.float32)
            scores = np.asarray(scores, dtype=np.float32)
            
            if kpts.ndim == 3:
                person_conf = scores.mean(axis=1)
                best_idx = int(np.argmax(person_conf))
                raw_kpts = kpts[best_idx]
                raw_scores = scores[best_idx]
            else:
                raw_kpts = kpts
                raw_scores = scores
                
        raw_kpts = np.asarray(raw_kpts, dtype=np.float32).reshape(17, 2)
        raw_scores = np.asarray(raw_scores, dtype=np.float32).reshape(17)
        
        # Apply Temporal Filter if enabled
        if temp_filter:
            raw_kpts, raw_scores, _ = temp_filter.filter_pose(raw_kpts, raw_scores)
            
        # Normalize
        norm_pose = normalizer.normalize(raw_kpts, raw_scores)
        
        kpts_list.append(raw_kpts)
        scores_list.append(raw_scores)
        norm_kpts_list.append(norm_pose.keypoints)
        
        frame_idx += 1
        if frame_idx % 100 == 0 or frame_idx >= total_frames:
            print(f"Processed {frame_idx}/{total_frames} frames...")
            
    cap.release()
    
    kpts_arr = np.stack(kpts_list).astype(np.float32)
    scores_arr = np.stack(scores_list).astype(np.float32)
    norm_kpts_arr = np.stack(norm_kpts_list).astype(np.float32)
    
    output_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz_path,
        keypoint=kpts_arr,
        keypoint_score=scores_arr,
        normalized_keypoint=norm_kpts_arr,
        label=np.array([label]),
        source_video=np.array([str(video_path)]),
    )
    print(f"Saved npz to {output_npz_path} with {len(kpts_list)} frames.")
    return True

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract poses from videos")
    parser.add_argument("--use-filter", action="store_true", help="Apply TemporalPoseFilter during extraction")
    parser.add_argument("--frame-skip", type=int, default=1, help="Frame skipping value (default: 1)")
    args = parser.parse_args()

    config_path = ROOT_DIR / "Edge" / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
        
    if 'detector' in config_data and 'onnx' in config_data['detector']:
        onnx_path = config_data['detector']['onnx'].get('model_path', '')
        if onnx_path.startswith('./'):
            config_data['detector']['onnx']['model_path'] = str(ROOT_DIR / "Edge" / onnx_path[2:])
            
    app_config = AppConfig(config_data)
    detector = create_detector(app_config.detector_config())
    normalizer = SpatialNormalizer(NormalizationParams(
        torso_confidence_threshold=0.5,
        allow_invalid_torso=True
    ))
    
    folder_suffix = ""
    if args.use_filter:
        folder_suffix += "_filter"
    if args.frame_skip > 1:
        folder_suffix += f"_skip{args.frame_skip}"
        
    folder_name_clips = f"clips{folder_suffix}" if folder_suffix else "clips"
    
    recorded_dir = ROOT_DIR / "Edge" / "Visualizar_Data" / "Data" / "Recorded"
    output_manifest_path = ROOT_DIR / "Edge" / "Visualizar_Data" / "Output" / folder_name_clips / "clips_manifest.jsonl"
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    
    targets = [
        ("Normal", "normal", "recorded_normal"),
        ("Shoplifting", "suspicious", "recorded_shoplifting")
    ]
    
    # Load existing manifest entries to avoid duplicates and skip already processed videos
    existing_entries = {}
    if output_manifest_path.exists():
        try:
            with open(output_manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        existing_entries[entry["npz_path"]] = entry
        except Exception as e:
            print(f"Warning: Could not read existing manifest file: {e}")
 
    for folder_name, label_val, output_subfolder in targets:
        folder_path = recorded_dir / folder_name
        if not folder_path.exists():
            print(f"Directory not found: {folder_path}")
            continue
            
        video_files = list(folder_path.glob("*.mp4")) + list(folder_path.glob("*.avi"))
        for video_path in video_files:
            npz_name = f"{video_path.stem}_processed.npz"
            output_npz_path = ROOT_DIR / "Edge" / "Visualizar_Data" / "Output" / folder_name_clips / output_subfolder / f"{video_path.stem}" / npz_name
            
            # Skip processing if NPZ file already exists
            npz_str = str(output_npz_path)
            if output_npz_path.exists():
                print(f"Skipping {video_path.name} (already processed: {npz_name})")
                # Keep it in manifest if not already registered
                if npz_str not in existing_entries:
                    existing_entries[npz_str] = {
                        "entry_id": f"clip_recorded_{video_path.stem}",
                        "video_path": str(video_path),
                        "npz_path": npz_str,
                        "label": label_val
                    }
                continue
                
            success = process_video(video_path, detector, normalizer, output_npz_path, label_val, use_filter=args.use_filter, frame_skip=args.frame_skip, app_config=app_config)
            if success:
                entry = {
                    "entry_id": f"clip_recorded_{video_path.stem}",
                    "video_path": str(video_path),
                    "npz_path": npz_str,
                    "label": label_val
                }
                existing_entries[npz_str] = entry
                print(f"Processed and registered: {video_path.name}")
                
    # Write the entire clean and deduplicated manifest back
    with open(output_manifest_path, "w", encoding="utf-8") as f:
        for entry in existing_entries.values():
            f.write(json.dumps(entry) + "\n")
            
    print("All processing done successfully.")

if __name__ == "__main__":
    main()
