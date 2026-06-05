#!/usr/bin/env python3
"""Runnable Phase 4 training entrypoint.

Loads a mix of manually cut clips and RetailS sequences, builds Phase 4
features, trains the attention LSTM, and saves the best checkpoint.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

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


def stratified_split(labels: np.ndarray, val_ratio: float = 0.2, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=np.int32)

    train_indices = []
    val_indices = []

    for cls in np.unique(labels):
        cls_indices = np.where(labels == cls)[0]
        rng.shuffle(cls_indices)
        val_count = max(1, int(round(len(cls_indices) * val_ratio))) if len(cls_indices) > 1 else 0
        val_indices.extend(cls_indices[:val_count].tolist())
        train_indices.extend(cls_indices[val_count:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return np.asarray(train_indices, dtype=np.int64), np.asarray(val_indices, dtype=np.int64)


def build_loaders(samples, batch_size: int, seed: int = 42, augment: bool = False):
    base_dataset = Phase4PoseDataset(samples, augment=False)
    labels = base_dataset.labels.astype(np.int32)
    train_idx, val_idx = stratified_split(labels, val_ratio=0.2, seed=seed)

    if len(val_idx) == 0:
        val_idx = train_idx[: max(1, len(train_idx) // 5)]
        train_idx = train_idx[max(1, len(train_idx) // 5):]

    train_dataset = Phase4PoseDataset(samples, augment=augment)
    val_dataset = Phase4PoseDataset(samples, augment=False)

    train_subset = Subset(train_dataset, train_idx.tolist())
    val_subset = Subset(val_dataset, val_idx.tolist())

    train_labels = labels[train_idx]
    class_counts = np.bincount(train_labels, minlength=2).astype(np.float32)
    class_counts[class_counts == 0] = 1.0
    sample_weights = np.asarray([1.0 / class_counts[label] for label in train_labels], dtype=np.float32)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=max(len(sample_weights), batch_size),
        replacement=True,
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Phase 4 model from manual clips and RetailS data")
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--manual-count", type=int, default=4)
    parser.add_argument("--retail-normal-count", type=int, default=4)
    parser.add_argument("--retail-suspicious-count", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--attention-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--augment", action="store_true", help="Enable light geometric augmentation during training")
    parser.add_argument("--save-path", type=str, default=str(ROOT_DIR / "Edge" / "models" / "retails_clip_model_precision.pth"))
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

    print("[1/4] Building samples...")
    samples = build_phase4_samples(data_config)
    if not samples:
        print("No training samples found. Export clips first, then rerun this script.")
        return

    labels = np.asarray([sample.label for sample in samples], dtype=np.int32)
    suspicious = int((labels == 1).sum())
    normal = int((labels == 0).sum())
    print(f"Loaded {len(samples)} samples -> normal={normal}, suspicious={suspicious}")

    train_loader, val_loader = build_loaders(samples, batch_size=args.batch_size, seed=args.seed, augment=args.augment)

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
    print("[2/4] Training...")
    for epoch in range(1, config.epochs + 1):
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.evaluate(val_loader)
        print(f"Epoch {epoch:02d}/{config.epochs} | train={train_loss:.4f} | val={val_loss:.4f}")

        if val_loss <= best_val:
            best_val = val_loss
            trainer.save(str(save_path))

    print("[3/4] Saving best model...")
    print(f"Best validation loss: {best_val:.4f}")
    print(f"Saved checkpoint: {save_path}")

    print("[4/4] Done.")


if __name__ == "__main__":
    main()