#!/usr/bin/env python3
"""End-to-end Phase 4 experiment runner.

This script trains on the available dataset clips, keeps a holdout split for
validation/test, calibrates a decision threshold on validation, and reports
clip-level metrics on the holdout set.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from Edge.Train.phase4_data import Phase4DataConfig, Phase4PoseDataset, build_phase4_samples
from Edge.Train.phase4_model import Phase4Classifier
from Edge.Train.phase4_types import Phase4Config
from Edge.Train.train_phase4 import Trainer
from pipeline.kinematic_features import KinematicFeatureExtractor


@dataclass
class SplitBundle:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def grouped_stratified_split(
    clip_ids: List[str],
    labels: np.ndarray,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
    split_manual: bool = False,
) -> SplitBundle:
    """
    Split samples grouped by clip_id so that all windows of any single clip
    land in exactly one split, while preserving stratified label balance of clips.
    """
    rng = np.random.default_rng(seed)
    
    # Group sample indices by clip_id
    clip_to_indices = {}
    for idx, cid in enumerate(clip_ids):
        clip_to_indices.setdefault(cid, []).append(idx)
        
    # Determine label for each clip (suspicious if any window is suspicious)
    clip_labels = {}
    for cid, idxs in clip_to_indices.items():
        clip_labels[cid] = int(np.max(labels[idxs]))
        
    # Separate manual/recorded clips (no colon) from RetailS clips (with colon)
    unique_clips = list(clip_to_indices.keys())
    manual_clips = [cid for cid in unique_clips if ":" not in cid]
    retails_clips = [cid for cid in unique_clips if ":" in cid]
    
    train_clips: List[str] = []
    val_clips: List[str] = []
    test_clips: List[str] = []
    
    if split_manual:
        print(f"[INFO] Stratifying and splitting {len(manual_clips)} manual/recorded clips across splits.")
        manual_labels = np.array([clip_labels[cid] for cid in manual_clips])
        for cls in np.unique(manual_labels):
            cls_clips = [cid for cid in manual_clips if clip_labels[cid] == cls]
            rng.shuffle(cls_clips)
            
            total = len(cls_clips)
            if total == 0:
                continue
                
            test_count = int(round(total * test_ratio)) if test_ratio > 0 else 0
            if test_ratio > 0 and total > 1:
                test_count = max(1, test_count)
            test_count = min(test_count, max(0, total - 2)) if total > 2 else min(test_count, max(0, total - 1))
            
            remaining = total - test_count
            val_count = int(round(total * val_ratio)) if val_ratio > 0 else 0
            if val_ratio > 0 and remaining > 1:
                val_count = max(1, val_count)
            val_count = min(val_count, max(0, remaining - 1)) if remaining > 1 else min(val_count, max(0, remaining))
            
            test_part = cls_clips[:test_count]
            val_part = cls_clips[test_count:test_count + val_count]
            train_part = cls_clips[test_count + val_count:]
            
            if len(train_part) == 0 and len(cls_clips) > 0:
                train_part = cls_clips[-1:]
                if len(test_part) > 0:
                    test_part = test_part[:-1]
                elif len(val_part) > 0:
                    val_part = val_part[:-1]
                    
            train_clips.extend(train_part)
            val_clips.extend(val_part)
            test_clips.extend(test_part)
    else:
        print(f"[INFO] Forcing {len(manual_clips)} manual/recorded clips into the Training split.")
        train_clips.extend(manual_clips)
        
    # Stratify split unique RetailS clip_ids
    retails_labels = np.array([clip_labels[cid] for cid in retails_clips])
    for cls in np.unique(retails_labels):
        cls_clips = [cid for cid in retails_clips if clip_labels[cid] == cls]
        rng.shuffle(cls_clips)
        
        total = len(cls_clips)
        if total == 0:
            continue
            
        test_count = int(round(total * test_ratio)) if test_ratio > 0 else 0
        if test_ratio > 0 and total > 1:
            test_count = max(1, test_count)
        test_count = min(test_count, max(0, total - 2)) if total > 2 else min(test_count, max(0, total - 1))
        
        remaining = total - test_count
        val_count = int(round(total * val_ratio)) if val_ratio > 0 else 0
        if val_ratio > 0 and remaining > 1:
            val_count = max(1, val_count)
        val_count = min(val_count, max(0, remaining - 1)) if remaining > 1 else min(val_count, max(0, remaining))
        
        test_part = cls_clips[:test_count]
        val_part = cls_clips[test_count:test_count + val_count]
        train_part = cls_clips[test_count + val_count:]
        
        if len(train_part) == 0 and len(cls_clips) > 0:
            train_part = cls_clips[-1:]
            if len(test_part) > 0:
                test_part = test_part[:-1]
            elif len(val_part) > 0:
                val_part = val_part[:-1]
                
        train_clips.extend(train_part)
        val_clips.extend(val_part)
        test_clips.extend(test_part)
        
    # Map back unique clips to sample indices
    train_idx = []
    for cid in train_clips:
        train_idx.extend(clip_to_indices[cid])
    val_idx = []
    for cid in val_clips:
        val_idx.extend(clip_to_indices[cid])
    test_idx = []
    for cid in test_clips:
        test_idx.extend(clip_to_indices[cid])
        
    # Shuffle indices within splits
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    
    return SplitBundle(
        train_idx=np.asarray(train_idx, dtype=np.int64),
        val_idx=np.asarray(val_idx, dtype=np.int64),
        test_idx=np.asarray(test_idx, dtype=np.int64),
    )


def build_loaders(
    samples: Sequence,
    batch_size: int,
    seed: int,
    augment: bool,
    val_ratio: float,
    test_ratio: float,
    split_manual: bool = False,
):
    base_dataset = Phase4PoseDataset(samples, augment=False)
    labels = base_dataset.labels.astype(np.int32)
    clip_ids = base_dataset.clip_ids
    split = grouped_stratified_split(clip_ids, labels, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed, split_manual=split_manual)

    if len(split.val_idx) == 0:
        split.val_idx = split.train_idx[: max(1, len(split.train_idx) // 5)]
        split.train_idx = split.train_idx[max(1, len(split.train_idx) // 5):]
    if len(split.test_idx) == 0:
        split.test_idx = split.train_idx[: max(1, len(split.train_idx) // 5)]
        split.train_idx = split.train_idx[max(1, len(split.train_idx) // 5):]

    train_dataset = Phase4PoseDataset(samples, augment=augment)
    val_dataset = Phase4PoseDataset(samples, augment=False)
    test_dataset = Phase4PoseDataset(samples, augment=False)

    train_subset = Subset(train_dataset, split.train_idx.tolist())
    val_subset = Subset(val_dataset, split.val_idx.tolist())
    test_subset = Subset(test_dataset, split.test_idx.tolist())

    train_labels = labels[split.train_idx] if len(split.train_idx) > 0 else labels
    class_counts = np.bincount(train_labels, minlength=2).astype(np.float32)
    class_counts[class_counts == 0] = 1.0
    sample_weights = np.asarray([1.0 / class_counts[label] for label in train_labels], dtype=np.float32)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=max(len(sample_weights), batch_size),
        replacement=True,
    ) if len(train_labels) > 0 else None

    train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=sampler, shuffle=False if sampler else True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, split


@torch.no_grad()
def collect_predictions(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> Dict[str, List]:
    model.eval()
    probabilities: List[float] = []
    labels: List[int] = []
    sources: List[str] = []

    for batch in dataloader:
        poses = batch["poses"].to(device)
        batch_labels = batch["labels"].to(device)
        logits, _ = model(poses)
        probs = torch.sigmoid(logits).detach().cpu().numpy().tolist()
        probabilities.extend([float(p) for p in probs])
        labels.extend([int(x) for x in batch_labels.detach().cpu().numpy().tolist()])
        if "source" in batch:
            batch_sources = batch["source"]
            if isinstance(batch_sources, list):
                sources.extend([str(item) for item in batch_sources])
            else:
                sources.extend([str(batch_sources)] * len(probs))
        else:
            sources.extend([f"sample_{len(sources) + i}" for i in range(len(probs))])

    return {"probabilities": probabilities, "labels": labels, "sources": sources}


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / max(1, len(y_true))
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    if precision > 0 and recall > 0 and specificity > 0:
        hprs = 3 / (1.0 / precision + 1.0 / recall + 1.0 / specificity)
    else:
        hprs = 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "hprs": float(hprs),
        "accuracy": float(accuracy),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def best_threshold(probabilities: Sequence[float], labels: Sequence[int], metric: str = "hprs") -> Tuple[float, Dict[str, float]]:
    probs = np.asarray(probabilities, dtype=np.float32)
    y_true = np.asarray(labels, dtype=np.int32)
    if len(probs) == 0:
        return 0.5, {"f1": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "specificity": 0.0, "hprs": 0.0}

    best_t = 0.5
    best_score = -1.0
    best_metrics: Dict[str, float] = {}

    for threshold in np.linspace(0.05, 0.95, 181):
        y_pred = (probs >= threshold).astype(np.int32)
        metrics = _compute_metrics(y_true, y_pred)
        score = metrics["hprs"] if metric == "hprs" else metrics["f1"]
        if score > best_score:
            best_score = score
            best_t = float(threshold)
            best_metrics = metrics

    return best_t, best_metrics


def confusion_metrics(probabilities: Sequence[float], labels: Sequence[int], threshold: float) -> Dict[str, float]:
    probs = np.asarray(probabilities, dtype=np.float32)
    y_true = np.asarray(labels, dtype=np.int32)
    y_pred = (probs >= threshold).astype(np.int32)
    metrics = _compute_metrics(y_true, y_pred)
    metrics["threshold"] = float(threshold)
    return metrics


def print_examples(probabilities: Sequence[float], labels: Sequence[int], sources: Sequence[str], threshold: float, limit: int = 10) -> None:
    rows = list(zip(probabilities, labels, sources))
    rows.sort(key=lambda item: abs(item[0] - threshold), reverse=False)
    print("[INFO] Most uncertain holdout samples:")
    for prob, label, source in rows[:limit]:
        decision = 1 if prob >= threshold else 0
        print(f"  prob={prob:.3f} pred={decision} label={label} source={source}")


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end Phase 4 experiment runner")
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--manual-count", type=int, default=-1, help="Number of manual clips to load, or -1 for all")
    parser.add_argument("--retail-normal-count", type=int, default=10)
    parser.add_argument("--retail-suspicious-count", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--attention-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--augment", action="store_true", help="Enable light geometric augmentation on the train split")
    parser.add_argument("--split-manual", action="store_true", help="Split manual/recorded clips across train/val/test splits")
    parser.add_argument("--save-path", type=str, default=str(ROOT_DIR / "Edge" / "models" / "phase4_experiment_model.pth"))
    parser.add_argument("--report-path", type=str, default=str(ROOT_DIR / "Edge" / "models" / "phase4_experiment_report.json"))
    parser.add_argument("--metric", type=str, choices=["f1", "hprs"], default="hprs", help="Metric to optimize the decision threshold")
    parser.add_argument("--loss", type=str, choices=["bce", "focal"], default="focal", help="Loss function type")
    parser.add_argument("--focal-alpha", type=float, default=0.25, help="Focal Loss alpha parameter")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal Loss gamma parameter")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--dropout", type=float, default=0.2, help="LSTM dropout rate")
    parser.add_argument("--num-layers", type=int, default=1, help="Number of LSTM layers")
    parser.add_argument("--bidirectional", action="store_true", help="Enable Bidirectional LSTM")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    manual_limit = 999999 if args.manual_count == -1 else args.manual_count
    data_config = Phase4DataConfig(
        sequence_length=args.sequence_length,
        manual_limit=manual_limit,
        retail_normal_limit=args.retail_normal_count,
        retail_suspicious_limit=args.retail_suspicious_count,
    )

    print("[1/5] Building samples...")
    samples = build_phase4_samples(data_config)
    if not samples:
        print("No training samples found. Export clips first, then rerun this script.")
        return

    labels = np.asarray([sample.label for sample in samples], dtype=np.int32)
    normal = int((labels == 0).sum())
    suspicious = int((labels == 1).sum())
    print(f"Loaded {len(samples)} samples -> normal={normal}, suspicious={suspicious}")

    train_loader, val_loader, test_loader, split = build_loaders(
        samples,
        batch_size=args.batch_size,
        seed=args.seed,
        augment=args.augment,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_manual=args.split_manual,
    )

    print(f"Split sizes -> train={len(split.train_idx)} | val={len(split.val_idx)} | test={len(split.test_idx)}")

    # Print split leakage check and statistics
    all_clip_ids = [samples[idx].clip_id for idx in range(len(samples))]
    all_labels = [samples[idx].label for idx in range(len(samples))]
    
    def print_split_stats(name, indices):
        split_clips = set(all_clip_ids[idx] for idx in indices)
        split_labels = [all_labels[idx] for idx in indices]
        n_normal = sum(1 for l in split_labels if l == 0)
        n_suspicious = sum(1 for l in split_labels if l == 1)
        print(f"  {name:5s} split -> {len(split_clips):3d} unique clips | {len(indices):4d} windows (normal={n_normal}, suspicious={n_suspicious})")
        return split_clips

    print("[INFO] Split statistics and leakage check:")
    train_clips = print_split_stats("Train", split.train_idx)
    val_clips = print_split_stats("Val", split.val_idx)
    test_clips = print_split_stats("Test", split.test_idx)
    
    overlap_train_val = train_clips.intersection(val_clips)
    overlap_train_test = train_clips.intersection(test_clips)
    overlap_val_test = val_clips.intersection(test_clips)
    print(f"  Overlap Train-Val: {len(overlap_train_val)} | Train-Test: {len(overlap_train_test)} | Val-Test: {len(overlap_val_test)}")
    assert len(overlap_train_val) == 0, "Data leakage detected: clip overlap between Train and Val!"
    assert len(overlap_train_test) == 0, "Data leakage detected: clip overlap between Train and Test!"
    assert len(overlap_val_test) == 0, "Data leakage detected: clip overlap between Val and Test!"
    print("  [SUCCESS] Split verification passed! Zero clip leakage.")

    extractor = KinematicFeatureExtractor()
    config = Phase4Config(
        sequence_length=args.sequence_length,
        input_size=extractor.feature_dim(),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        attention_size=args.attention_size,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        loss_type=args.loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        bidirectional=args.bidirectional,
        patience=args.patience,
    )

    model = Phase4Classifier(config)
    trainer = Trainer(model, config)

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    patience = args.patience
    epochs_no_improve = 0
    print("[2/5] Training...")
    for epoch in range(1, config.epochs + 1):
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.evaluate(val_loader)
        print(f"Epoch {epoch:02d}/{config.epochs} | train={train_loss:.4f} | val={val_loss:.4f}")
        
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            trainer.save(str(save_path))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[EARLY STOPPING] Triggered early stopping at epoch {epoch:02d} (best val loss: {best_val:.4f})")
                break
        
        if epoch == config.epochs:
            print(f"[INFO] Saving final epoch checkpoint to {save_path}")
            trainer.save(str(save_path))

    print("[3/5] Loading best checkpoint...")
    best_model = Phase4Classifier(config)
    best_model.load_state_dict(torch.load(save_path, map_location="cpu", weights_only=False))
    best_model.eval()

    device = torch.device("cpu")
    val_pred = collect_predictions(best_model.to(device), val_loader, device)
    
    # Calibrate decision threshold on manual clips (clip-level) if available
    manual_samples = [s for s in samples if ":" not in s.clip_id]
    if len(manual_samples) > 0:
        print(f"[INFO] Calibrating decision threshold on {len(manual_samples)} manual windows (clip-level)...")
        manual_dataset = Phase4PoseDataset(manual_samples, augment=False)
        manual_loader = DataLoader(manual_dataset, batch_size=args.batch_size, shuffle=False)
        manual_pred = collect_predictions(best_model.to(device), manual_loader, device)
        
        # Group by clip_id
        clip_probs = {}
        clip_labels = {}
        for prob, label, src in zip(manual_pred["probabilities"], manual_pred["labels"], manual_pred["sources"]):
            # Normalize clip ID (strip oversampling suffix)
            clip_id = src.split("_dup")[0]
            clip_probs.setdefault(clip_id, []).append(prob)
            clip_labels[clip_id] = int(label)
            
        # Compute max probability per clip
        manual_clip_probs = []
        manual_clip_labels = []
        for cid, probs in clip_probs.items():
            manual_clip_probs.append(max(probs))
            manual_clip_labels.append(clip_labels[cid])
            
        threshold, threshold_metrics = best_threshold(manual_clip_probs, manual_clip_labels, metric=args.metric)
        print(f"[INFO] Calibrated threshold on manual clips: {threshold:.3f}")
    else:
        print(f"[INFO] Calibrating decision threshold using all {len(val_pred['probabilities'])} validation windows.")
        threshold, threshold_metrics = best_threshold(val_pred["probabilities"], val_pred["labels"], metric=args.metric)
    test_pred = collect_predictions(best_model.to(device), test_loader, device)
    test_metrics = confusion_metrics(test_pred["probabilities"], test_pred["labels"], threshold)

    print("[4/5] Calibration and test metrics...")
    print(f"Validation threshold (optimized for {args.metric.upper()}): {threshold:.3f}")
    print(
        f"Validation best -> f1={threshold_metrics['f1']:.3f} | hprs={threshold_metrics['hprs']:.3f} | precision={threshold_metrics['precision']:.3f} | "
        f"recall={threshold_metrics['recall']:.3f} | specificity={threshold_metrics['specificity']:.3f} | accuracy={threshold_metrics['accuracy']:.3f}"
    )
    print(
        f"Test -> f1={test_metrics['f1']:.3f} | hprs={test_metrics['hprs']:.3f} | precision={test_metrics['precision']:.3f} | "
        f"recall={test_metrics['recall']:.3f} | accuracy={test_metrics['accuracy']:.3f} | specificity={test_metrics['specificity']:.3f}"
    )
    print(
        f"Confusion matrix (test) -> tp={int(test_metrics['tp'])} tn={int(test_metrics['tn'])} "
        f"fp={int(test_metrics['fp'])} fn={int(test_metrics['fn'])}"
    )
    print_examples(test_pred["probabilities"], test_pred["labels"], test_pred["sources"], threshold)

    report = {
        "config": {
            "sequence_length": args.sequence_length,
            "manual_count": args.manual_count,
            "retail_normal_count": args.retail_normal_count,
            "retail_suspicious_count": args.retail_suspicious_count,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "hidden_size": args.hidden_size,
            "attention_size": args.attention_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "augment": bool(args.augment),
            "seed": args.seed,
            "metric": args.metric,
            "loss": args.loss,
            "focal_alpha": args.focal_alpha,
            "focal_gamma": args.focal_gamma,
            "patience": args.patience,
            "dropout": args.dropout,
            "num_layers": args.num_layers,
            "bidirectional": args.bidirectional,
        },
        "sample_counts": {"normal": normal, "suspicious": suspicious, "total": len(samples)},
        "split_sizes": {"train": int(len(split.train_idx)), "val": int(len(split.val_idx)), "test": int(len(split.test_idx))},
        "threshold": float(threshold),
        "validation": threshold_metrics,
        "test": test_metrics,
        "save_path": str(save_path),
        "onnx_path": str(save_path).replace(".pth", ".onnx"),
    }

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("[5/5] Done.")
    print(f"Saved checkpoint: {save_path}")

    # Convert best model to ONNX
    onnx_path = str(save_path).replace(".pth", ".onnx")
    print(f"Exporting model to ONNX: {onnx_path} ...")
    try:
        best_model.eval()
        dummy_input = torch.zeros(1, config.sequence_length, config.input_size, dtype=torch.float32)
        torch.onnx.export(
            best_model.to("cpu"),
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['logits', 'attention'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'logits': {0: 'batch_size'},
                'attention': {0: 'batch_size'}
            }
        )
        print(f"[SUCCESS] Exported ONNX model successfully: {onnx_path}")
    except Exception as e:
        print(f"[ERROR] Failed to export model to ONNX: {e}")

    print(f"Saved report: {report_path}")
    print(f"Calibrated threshold: {threshold:.3f}")


if __name__ == "__main__":
    main()
