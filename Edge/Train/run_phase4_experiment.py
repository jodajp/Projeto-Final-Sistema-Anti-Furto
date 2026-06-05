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


def stratified_split_three(
    labels: np.ndarray,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> SplitBundle:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=np.int32)

    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []

    for cls in np.unique(labels):
        cls_indices = np.where(labels == cls)[0]
        rng.shuffle(cls_indices)
        total = len(cls_indices)
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

        test_part = cls_indices[:test_count]
        val_part = cls_indices[test_count:test_count + val_count]
        train_part = cls_indices[test_count + val_count:]

        if len(train_part) == 0 and len(cls_indices) > 0:
            train_part = cls_indices[-1:]
            if len(test_part) > 0:
                test_part = test_part[:-1]
            elif len(val_part) > 0:
                val_part = val_part[:-1]

        train_indices.extend(train_part.tolist())
        val_indices.extend(val_part.tolist())
        test_indices.extend(test_part.tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)
    return SplitBundle(
        train_idx=np.asarray(train_indices, dtype=np.int64),
        val_idx=np.asarray(val_indices, dtype=np.int64),
        test_idx=np.asarray(test_indices, dtype=np.int64),
    )


def build_loaders(
    samples: Sequence,
    batch_size: int,
    seed: int,
    augment: bool,
    val_ratio: float,
    test_ratio: float,
):
    base_dataset = Phase4PoseDataset(samples, augment=False)
    labels = base_dataset.labels.astype(np.int32)
    split = stratified_split_three(labels, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)

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


def best_threshold(probabilities: Sequence[float], labels: Sequence[int]) -> Tuple[float, Dict[str, float]]:
    probs = np.asarray(probabilities, dtype=np.float32)
    y_true = np.asarray(labels, dtype=np.int32)
    if len(probs) == 0:
        return 0.5, {"f1": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0}

    best_t = 0.5
    best_f1 = -1.0
    best_metrics: Dict[str, float] = {}

    for threshold in np.linspace(0.05, 0.95, 181):
        y_pred = (probs >= threshold).astype(np.int32)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / max(1, len(y_true))

        if f1 > best_f1:
            best_f1 = f1
            best_t = float(threshold)
            best_metrics = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "accuracy": float(accuracy),
                "tp": float(tp),
                "tn": float(tn),
                "fp": float(fp),
                "fn": float(fn),
            }

    return best_t, best_metrics


def confusion_metrics(probabilities: Sequence[float], labels: Sequence[int], threshold: float) -> Dict[str, float]:
    probs = np.asarray(probabilities, dtype=np.float32)
    y_true = np.asarray(labels, dtype=np.int32)
    y_pred = (probs >= threshold).astype(np.int32)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / max(1, len(y_true))
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "specificity": float(specificity),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


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
    parser.add_argument("--manual-count", type=int, default=2)
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
    parser.add_argument("--save-path", type=str, default=str(ROOT_DIR / "Edge" / "models" / "phase4_experiment_model.pth"))
    parser.add_argument("--report-path", type=str, default=str(ROOT_DIR / "Edge" / "models" / "phase4_experiment_report.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_config = Phase4DataConfig(
        sequence_length=args.sequence_length,
        manual_limit=args.manual_count,
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
    )

    print(f"Split sizes -> train={len(split.train_idx)} | val={len(split.val_idx)} | test={len(split.test_idx)}")

    extractor = KinematicFeatureExtractor()
    config = Phase4Config(
        sequence_length=args.sequence_length,
        input_size=extractor.feature_dim(),
        hidden_size=args.hidden_size,
        attention_size=args.attention_size,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
    )

    model = Phase4Classifier(config)
    trainer = Trainer(model, config)

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    print("[2/5] Training...")
    for epoch in range(1, config.epochs + 1):
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.evaluate(val_loader)
        print(f"Epoch {epoch:02d}/{config.epochs} | train={train_loss:.4f} | val={val_loss:.4f}")
        if val_loss <= best_val:
            best_val = val_loss
            trainer.save(str(save_path))

    print("[3/5] Loading best checkpoint...")
    best_model = Phase4Classifier(config)
    best_model.load_state_dict(torch.load(save_path, map_location="cpu", weights_only=False))
    best_model.eval()

    device = torch.device("cpu")
    val_pred = collect_predictions(best_model.to(device), val_loader, device)
    threshold, threshold_metrics = best_threshold(val_pred["probabilities"], val_pred["labels"])
    test_pred = collect_predictions(best_model.to(device), test_loader, device)
    test_metrics = confusion_metrics(test_pred["probabilities"], test_pred["labels"], threshold)

    print("[4/5] Calibration and test metrics...")
    print(f"Validation threshold: {threshold:.3f}")
    print(
        f"Validation best -> f1={threshold_metrics['f1']:.3f} | precision={threshold_metrics['precision']:.3f} | "
        f"recall={threshold_metrics['recall']:.3f} | accuracy={threshold_metrics['accuracy']:.3f}"
    )
    print(
        f"Test -> f1={test_metrics['f1']:.3f} | precision={test_metrics['precision']:.3f} | "
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
        },
        "sample_counts": {"normal": normal, "suspicious": suspicious, "total": len(samples)},
        "split_sizes": {"train": int(len(split.train_idx)), "val": int(len(split.val_idx)), "test": int(len(split.test_idx))},
        "threshold": float(threshold),
        "validation": threshold_metrics,
        "test": test_metrics,
        "save_path": str(save_path),
    }

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("[5/5] Done.")
    print(f"Saved checkpoint: {save_path}")
    print(f"Saved report: {report_path}")
    print(f"Calibrated threshold: {threshold:.3f}")


if __name__ == "__main__":
    main()
