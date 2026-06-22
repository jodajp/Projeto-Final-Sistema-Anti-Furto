#!/usr/bin/env python3
"""
Model Benchmarking Script
=========================
Evaluates the trained ONNX/PyTorch model on all custom/recorded clips registered
in the manifest. Runs in fast mode (pre-extracted NPZ keypoints) to complete
in under a second.

Usage:
  .venv\Scripts\python.exe Edge\Train\run_benchmark.py
"""
import sys
import os
import json
from pathlib import Path
import numpy as np
import yaml

# Setup path
EDGE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EDGE_DIR))

from pipeline.kinematic_features import KinematicFeatureExtractor

def load_onnx_session(onnx_path: Path):
    import onnxruntime as ort
    # Configure optimized options
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=['CPUExecutionProvider'])
    return session

def run_onnx_inference(session, features: np.ndarray) -> np.ndarray:
    ort_inputs = {session.get_inputs()[0].name: features.astype(np.float32)}
    ort_outs = session.run(None, ort_inputs)
    logits = ort_outs[0].flatten()
    probs = 1.0 / (1.0 + np.exp(-logits))  # Sigmoid
    return probs

def load_pytorch_model(model_path: Path, config_dict: dict, device: str = "cpu"):
    import torch
    from Train.phase4_model import Phase4Classifier
    from Train.phase4_types import Phase4Config
    
    config = Phase4Config(
        sequence_length=config_dict.get("sequence_length", 45),
        input_size=config_dict.get("input_size", 60),
        hidden_size=128,
        num_layers=1,
        attention_size=64,
        dropout=0.1
    )
    model = Phase4Classifier(config).to(device)
    checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

def run_pytorch_inference(model, features: np.ndarray, device: str = "cpu") -> np.ndarray:
    import torch
    features_tensor = torch.from_numpy(features).float().to(device)
    with torch.no_grad():
        logits, _ = model(features_tensor)
    probs = torch.sigmoid(logits).cpu().numpy().flatten()
    return probs

def main():
    print("=" * 60)
    print("           ANTI-THEFT SYSTEM: MODEL BENCHMARK           ")
    print("=" * 60)
    
    # 1. Load Calibration Report & Config
    report_path = EDGE_DIR / "models" / "phase4_experiment_report.json"
    config_path = EDGE_DIR / "config.yaml"
    
    threshold = 0.40  # Default fallback
    sequence_length = 45
    onnx_path = EDGE_DIR / "models" / "phase4_experiment_model.onnx"
    pth_path = EDGE_DIR / "models" / "phase4_experiment_model.pth"
    
    if report_path.exists():
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            threshold = float(report.get("threshold", threshold))
            sequence_length = int(report.get("config", {}).get("sequence_length", sequence_length))
            print(f"[OK] Loaded calibrated threshold: {threshold:.3f} and sequence length: {sequence_length}")
        except Exception as e:
            print(f"[WARNING] Could not load report file: {e}. Using default threshold: {threshold}")
            
    # 2. Initialize Model Session (ONNX preferred, PyTorch fallback)
    model_type = "ONNX"
    session = None
    py_model = None
    
    if onnx_path.exists():
        try:
            session = load_onnx_session(onnx_path)
            print(f"[OK] Loaded ONNX model from {onnx_path.name}")
        except Exception as e:
            print(f"[WARNING] ONNX initialization failed: {e}. Trying PyTorch...")
            session = None
            
    if session is None:
        if pth_path.exists():
            try:
                # Load input size from extractor dim
                extractor = KinematicFeatureExtractor()
                config_dict = {"sequence_length": sequence_length, "input_size": extractor.feature_dim()}
                py_model = load_pytorch_model(pth_path, config_dict)
                model_type = "PyTorch"
                print(f"[OK] Loaded PyTorch checkpoint from {pth_path.name}")
            except Exception as e:
                print(f"[ERROR] PyTorch initialization failed: {e}")
                sys.exit(1)
        else:
            print("[ERROR] No trained models found. Train a model first using run_phase4_experiment.py.")
            sys.exit(1)
            
    # 3. Load Manifest Entries
    manifest_path = EDGE_DIR / "Visualizar_Data" / "Output" / "clips" / "clips_manifest.jsonl"
    if not manifest_path.exists():
        print(f"[ERROR] Manifest file not found: {manifest_path}. Please extract poses first.")
        sys.exit(1)
        
    manifest_entries = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                manifest_entries.append(json.loads(line))
                
    # Filter only custom/recorded/manual clips and deduplicate
    custom_entries = []
    seen_paths = set()
    for entry in manifest_entries:
        npz_path = entry.get("npz_path", "")
        if not npz_path or npz_path in seen_paths:
            continue
        entry_id = entry.get("entry_id", "")
        video_path = entry.get("video_path", "")
        # A clip is custom if it starts with clip_recorded, clip_shoplifting, or contains Recorded/Shoplifting in paths
        is_recorded = "recorded" in entry_id or "Recorded" in video_path
        is_manual = "shoplifting-" in entry_id or "shoplifting-" in video_path or "Shoplifting\\" in video_path
        if is_recorded or is_manual:
            custom_entries.append(entry)
            seen_paths.add(npz_path)
            
    if not custom_entries:
        print("[WARNING] No custom/recorded clips registered in the manifest yet.")
        print("Please extract poses of your recorded clips using extract_recorded_clips.py first.")
        sys.exit(0)
        
    print(f"[INFO] Found {len(custom_entries)} unique custom/recorded videos in manifest. Benchmarking...")
    print("-" * 80)
    
    extractor = KinematicFeatureExtractor()
    
    results = []
    y_true = []
    y_pred = []
    
    for entry in custom_entries:
        npz_path = Path(entry["npz_path"])
        label_str = entry.get("label", "suspicious").lower()
        true_label = 1 if label_str == "suspicious" else 0
        
        if not npz_path.exists():
            print(f"[SKIP] Path does not exist: {npz_path.name}")
            continue
            
        # Load keypoints
        with np.load(npz_path, allow_pickle=False) as data:
            kpt_key = "normalized_keypoint" if "normalized_keypoint" in data else "keypoint"
            coords = np.asarray(data[kpt_key], dtype=np.float32)
            
        if coords.ndim == 4:
            coords = coords[0]
            
        total_frames = len(coords)
        if total_frames < sequence_length:
            pad = sequence_length - total_frames
            coords_padded = np.pad(coords, ((0, pad), (0, 0), (0, 0)), mode="edge")
            windows = [coords_padded]
        else:
            windows = []
            for start in range(0, total_frames - sequence_length + 1):
                windows.append(coords[start:start + sequence_length])
                
        windows_arr = np.stack(windows, axis=0)  # (W, T, 17, 2)
        features = extractor.transform(windows_arr)  # (W, T, D)
        
        # Run inference
        if model_type == "ONNX":
            probs = run_onnx_inference(session, features)
        else:
            probs = run_pytorch_inference(py_model, features)
            
        max_prob = float(np.max(probs)) if len(probs) > 0 else 0.0
        predicted_label = 1 if max_prob >= threshold else 0
        
        y_true.append(true_label)
        y_pred.append(predicted_label)
        
        results.append({
            "name": npz_path.name.replace("_processed.npz", ".mp4"),
            "true_label": "SHOPLIFTING" if true_label == 1 else "NORMAL",
            "pred_label": "SHOPLIFTING" if predicted_label == 1 else "NORMAL",
            "max_prob": max_prob,
            "status": "PASS" if true_label == predicted_label else "FAIL"
        })
        
    # 4. Print Results Table
    print(f"{'VIDEO FILE':40s} | {'GROUND TRUTH':12s} | {'MAX PROB':8s} | {'PREDICTION':12s} | {'STATUS':5s}")
    print("-" * 85)
    for res in results:
        status_color = "[OK]" if res["status"] == "PASS" else "[X]"
        print(f"{res['name']:40s} | {res['true_label']:12s} | {res['max_prob']:8.4f} | {res['pred_label']:12s} | {res['status']:5s}")
        
    print("-" * 85)
    
    # 5. Compute Metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    total = len(y_true)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    print("\nBENCHMARK METRICS:")
    print(f"  Total Custom Videos : {total}")
    print(f"  Correct Predictions  : {tp + tn} / {total} (Passed)")
    print(f"  Accuracy             : {accuracy * 100:.1f}%")
    print(f"  Precision            : {precision * 100:.1f}%")
    print(f"  Recall (Sensitivity) : {recall * 100:.1f}%")
    print(f"  Specificity          : {specificity * 100:.1f}%")
    print(f"  F1-Score             : {f1 * 100:.1f}%")
    print(f"  Confusion Matrix     : TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print("=" * 60)

if __name__ == "__main__":
    main()
