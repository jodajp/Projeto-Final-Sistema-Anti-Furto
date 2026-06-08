from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.kinematic_features import KinematicFeatureExtractor
from pipeline.spatial_normalizer import NormalizationParams, SpatialNormalizer


@dataclass
class Phase4DataConfig:
    sequence_length: int = 30
    manual_manifest_path: Path = ROOT_DIR / "Visualizar_Data" / "Output" / "clips" / "clips_manifest.jsonl"
    retail_normal_limit: int = 4
    retail_suspicious_limit: int = 4
    manual_limit: int = 4


@dataclass
class Phase4Sample:
    coords: np.ndarray  # (T, 17, 2)
    scores: np.ndarray  # (T, 17)
    label: int
    source: str = ""
    clip_id: str = ""


def _select_window(coords: np.ndarray, scores: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    total = int(coords.shape[0])
    if total == 0:
        raise ValueError("Empty sequence")

    if total <= sequence_length:
        pad = sequence_length - total
        if pad > 0:
            coords = np.pad(coords, ((0, pad), (0, 0), (0, 0)), mode="edge")
            scores = np.pad(scores, ((0, pad), (0, 0)), mode="edge")
        return coords[:sequence_length], scores[:sequence_length]

    wrists = coords[:, [9, 10], :].mean(axis=1)
    hips = coords[:, [11, 12], :].mean(axis=1)
    shoulders = coords[:, [5, 6], :].mean(axis=1)

    hip_valid = ~np.isnan(hips).any(axis=1)
    shoulder_valid = ~np.isnan(shoulders).any(axis=1)
    hip_center = np.where(hip_valid[:, None], hips, shoulders)
    fallback_center = np.where(shoulder_valid[:, None], shoulders, wrists)
    hip_center = np.where(np.isnan(hip_center).any(axis=1, keepdims=True), fallback_center, hip_center)

    dist = np.linalg.norm(wrists - hip_center, axis=1)
    dist = np.nan_to_num(dist, nan=np.inf, posinf=np.inf, neginf=np.inf)
    center = int(np.argmin(dist)) if np.isfinite(dist).any() else total // 2

    start = max(0, center - sequence_length // 2)
    end = start + sequence_length
    if end > total:
        end = total
        start = max(0, end - sequence_length)

    return coords[start:end], scores[start:end]


def _select_windows(
    coords: np.ndarray,
    scores: np.ndarray,
    sequence_length: int,
    max_windows: int = 3,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    total = int(coords.shape[0])
    if total <= sequence_length:
        return [_select_window(coords, scores, sequence_length)]

    last_start = total - sequence_length
    if max_windows <= 1 or last_start <= 0:
        return [_select_window(coords, scores, sequence_length)]

    start_positions = np.linspace(0, last_start, num=min(max_windows, last_start + 1), dtype=int)
    windows: List[Tuple[np.ndarray, np.ndarray]] = []
    seen = set()
    for start in start_positions:
        start_idx = int(start)
        if start_idx in seen:
            continue
        seen.add(start_idx)
        end_idx = start_idx + sequence_length
        windows.append((coords[start_idx:end_idx], scores[start_idx:end_idx]))

    return windows or [_select_window(coords, scores, sequence_length)]


def _normalize_sequence(coords: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    normalizer = SpatialNormalizer(
        NormalizationParams(
            torso_confidence_threshold=0.5,
            min_torso_length_px=10.0,
            allow_invalid_torso=True,
        )
    )

    normalized_coords = []
    normalized_scores = []

    for frame_coords, frame_scores in zip(coords, scores):
        pose = normalizer.normalize(frame_coords, frame_scores)
        normalized_coords.append(pose.keypoints)
        normalized_scores.append(pose.scores)

    return np.asarray(normalized_coords, dtype=np.float32), np.asarray(normalized_scores, dtype=np.float32)


def _load_manual_samples(manifest_path: Path, sequence_length: int, limit: int) -> List[Phase4Sample]:
    samples: List[Phase4Sample] = []
    if not manifest_path.exists():
        return samples

    manifest_entries = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                manifest_entries.append(json.loads(line))

    if not manifest_entries:
        return samples

    for entry in manifest_entries[:limit]:
        npz_path = Path(entry["npz_path"])
        if not npz_path.exists():
            continue

        with np.load(npz_path, allow_pickle=False) as data:
            if "normalized_keypoint" in data:
                coords = np.asarray(data["normalized_keypoint"], dtype=np.float32)
            else:
                coords = np.asarray(data["keypoint"], dtype=np.float32)

            scores = np.asarray(data["keypoint_score"], dtype=np.float32)
            label = 1 if str(entry.get("label", "suspicious")).lower() == "suspicious" else 0

        if coords.ndim == 4:
            coords = coords[0]
        if scores.ndim == 3:
            scores = scores[0]

        for window_coords, window_scores in _select_windows(coords, scores, sequence_length):
            if window_coords.shape != (sequence_length, 17, 2) or window_scores.shape != (sequence_length, 17):
                continue
            source_name = npz_path.stem
            samples.append(Phase4Sample(coords=window_coords, scores=window_scores, label=label, source=source_name, clip_id=source_name))

    return samples


def build_phase4_samples(config: Phase4DataConfig) -> List[Phase4Sample]:
    samples: List[Phase4Sample] = []
    
    # Load manual clips
    if config.manual_limit > 0:
        manual_samples = _load_manual_samples(
            config.manual_manifest_path,
            config.sequence_length,
            config.manual_limit
        )
        # Dynamic class-balanced oversampling targeting 90 windows per class
        normal_samples = [s for s in manual_samples if s.label == 0]
        suspicious_samples = [s for s in manual_samples if s.label == 1]
        
        n_normal = len(normal_samples)
        n_suspicious = len(suspicious_samples)
        
        target_windows = 180
        oversampled_manual = []
        
        if n_normal > 0:
            dup_factor_normal = max(1, target_windows // n_normal)
            for sample in normal_samples:
                for i in range(dup_factor_normal):
                    oversampled_manual.append(
                        Phase4Sample(
                            coords=sample.coords.copy(),
                            scores=sample.scores.copy(),
                            label=sample.label,
                            source=f"{sample.source}_dup{i}",
                            clip_id=sample.clip_id,
                        )
                    )
                    
        if n_suspicious > 0:
            dup_factor_suspicious = max(1, target_windows // n_suspicious)
            for sample in suspicious_samples:
                for i in range(dup_factor_suspicious):
                    oversampled_manual.append(
                        Phase4Sample(
                            coords=sample.coords.copy(),
                            scores=sample.scores.copy(),
                            label=sample.label,
                            source=f"{sample.source}_dup{i}",
                            clip_id=sample.clip_id,
                        )
                    )
        samples.extend(oversampled_manual)

    # Load RetailS clips using mask-aware disk loader
    staged_limit = config.retail_suspicious_limit // 2 if config.retail_suspicious_limit else None
    realworld_limit = config.retail_suspicious_limit - staged_limit if config.retail_suspicious_limit and staged_limit is not None else None
    
    retail_samples = load_retails_disk_samples(
        sequence_length=config.sequence_length,
        normal_file_limit=config.retail_normal_limit,
        staged_file_limit=staged_limit,
        realworld_file_limit=realworld_limit,
    )
    samples.extend(retail_samples)

    return samples


class Phase4PoseDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Phase4Sample],
        augment: bool = False,
        augmentation_noise: float = 0.015,
        augmentation_scale: float = 0.04,
        augmentation_translate: float = 0.03,
    ) -> None:
        if not samples:
            raise ValueError("Phase4PoseDataset requires at least one sample")

        coords = np.stack([sample.coords for sample in samples]).astype(np.float32)
        scores = np.stack([sample.scores for sample in samples]).astype(np.float32)
        labels = np.asarray([sample.label for sample in samples], dtype=np.float32)

        extractor = KinematicFeatureExtractor()
        feats = extractor.transform(coords)

        self.coords = coords
        self.scores = scores
        self.feats = feats.astype(np.float32)
        self.labels = labels
        self.sources = [sample.source for sample in samples]
        self.clip_ids = [sample.clip_id for sample in samples]
        self.augment = bool(augment)
        self.augmentation_noise = float(augmentation_noise)
        self.augmentation_scale = float(augmentation_scale)
        self.augmentation_translate = float(augmentation_translate)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        coords = self.coords[idx]
        scores = self.scores[idx]

        if self.augment:
            rng = np.random.default_rng()
            augmented = coords.copy()

            scale = 1.0 + rng.normal(0.0, self.augmentation_scale)
            translate = rng.normal(0.0, self.augmentation_translate, size=(1, 1, 2)).astype(np.float32)
            jitter = rng.normal(0.0, self.augmentation_noise, size=augmented.shape).astype(np.float32)

            augmented = augmented * np.float32(scale)
            augmented = augmented + translate + jitter

            # Keep the augmentation conservative: the label should not change,
            # but the model sees slightly different spatial realizations.
            feats = KinematicFeatureExtractor().transform(augmented[np.newaxis, ...])[0]
            
            # Speed augmentation: multiply velocities (first 34 dims) by a random scale factor
            speed_factor = rng.uniform(0.7, 1.3)
            feats[:, :34] *= np.float32(speed_factor)
        else:
            feats = self.feats[idx].copy()

        return {
            "poses": torch.from_numpy(feats),
            "confidences": torch.from_numpy(scores),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
            "source": self.sources[idx],
            "clip_id": self.clip_ids[idx],
        }


def _load_pose_clip(json_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with open(json_path, "r", encoding="utf-8") as f:
        pose_data = json.load(f)

    if not pose_data:
        raise ValueError(f"Empty pose file: {json_path}")

    def _person_frame_count(person_frames: dict) -> int:
        return sum(1 for key in person_frames.keys() if str(key).isdigit())

    person_id, person_frames = max(pose_data.items(), key=lambda item: _person_frame_count(item[1]))
    frame_keys = sorted(int(k) for k in person_frames.keys() if str(k).isdigit())
    if not frame_keys:
        raise ValueError(f"No frames found in pose file: {json_path}")

    start_frame = frame_keys[0]
    end_frame = frame_keys[-1]

    coords_frames: List[np.ndarray] = []
    scores_frames: List[np.ndarray] = []

    for frame_idx in range(start_frame, end_frame + 1):
        frame_key = str(frame_idx)
        if frame_key not in person_frames:
            if coords_frames:
                coords_frames.append(coords_frames[-1])
                scores_frames.append(scores_frames[-1])
            continue

        raw = np.asarray(person_frames[frame_key]["keypoints"], dtype=np.float32)
        if raw.shape == (51,):
            coords = raw.reshape(17, 3)[:, :2]
            scores = raw.reshape(17, 3)[:, 2]
        elif raw.shape == (34,):
            coords = raw.reshape(17, 2)
            scores = np.ones((17,), dtype=np.float32)
        else:
            coords = raw.reshape(17, 2)
            scores = np.ones((17,), dtype=np.float32)

        coords_frames.append(coords)
        scores_frames.append(scores)

    coords = np.asarray(coords_frames, dtype=np.float32)
    scores = np.asarray(scores_frames, dtype=np.float32)
    if coords.ndim != 3 or coords.shape[1:] != (17, 2):
        raise ValueError(f"Unexpected coords shape in {json_path}: {coords.shape}")
    if scores.ndim != 2 or scores.shape[1] != 17:
        raise ValueError(f"Unexpected scores shape in {json_path}: {scores.shape}")

    return coords, scores


def _load_mask_clip(mask_path: Path, expected_length: int) -> np.ndarray:
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    mask = np.asarray(np.load(mask_path), dtype=np.uint8).reshape(-1)
    if len(mask) == expected_length:
        return mask
    if len(mask) < expected_length:
        pad_value = int(mask[-1]) if len(mask) else 0
        return np.pad(mask, (0, expected_length - len(mask)), mode="constant", constant_values=pad_value)
    return mask[:expected_length]


def _build_windows_from_clip(
    coords: np.ndarray,
    scores: np.ndarray,
    sequence_length: int,
    source: str,
    label: int,
    windows_per_clip: int = 3,
    mask: Optional[np.ndarray] = None,
) -> List[Phase4Sample]:
    total = int(coords.shape[0])
    if total == 0:
        return []

    if total <= sequence_length:
        windows = [_select_window(coords, scores, sequence_length)]
        start_positions = [0]
    else:
        last_start = total - sequence_length
        start_positions = np.linspace(0, last_start, num=min(windows_per_clip, last_start + 1), dtype=int)
        windows = []
        seen = set()
        for start in start_positions:
            start_idx = int(start)
            if start_idx in seen:
                continue
            seen.add(start_idx)
            end_idx = start_idx + sequence_length
            windows.append((coords[start_idx:end_idx], scores[start_idx:end_idx]))

        if not windows:
            windows = [_select_window(coords, scores, sequence_length)]
            start_positions = [max(0, total // 2 - sequence_length // 2)]
        else:
            start_positions = [int(start) for start in start_positions[: len(windows)]]

    samples: List[Phase4Sample] = []
    for idx, (window_coords, window_scores) in enumerate(windows):
        if window_coords.shape != (sequence_length, 17, 2) or window_scores.shape != (sequence_length, 17):
            continue
        window_coords, window_scores = _normalize_sequence(window_coords, window_scores)
        window_label = int(label)
        if mask is not None and len(mask) > 0:
            start_idx = start_positions[idx] if idx < len(start_positions) else 0
            end_idx = min(start_idx + sequence_length, len(mask))
            if end_idx > start_idx:
                window_label = int(np.any(mask[start_idx:end_idx] > 0))
        samples.append(
            Phase4Sample(
                coords=window_coords,
                scores=window_scores,
                label=window_label,
                source=f"{source}:{idx}",
                clip_id=source,
            )
        )

    return samples


def load_retails_disk_samples(
    sequence_length: int,
    normal_file_limit: Optional[int] = None,
    staged_file_limit: Optional[int] = None,
    realworld_file_limit: Optional[int] = None,
    windows_per_clip: int = 3,
) -> List[Phase4Sample]:
    """Load RetailS clips directly from the local dataset tree.

    Normal clips come from RetailS_train and are labeled 0.
    Staged and realworld clips use the matching frame-mask .npy file to
    derive clip windows labeled 1 whenever any frame in the window is anomalous.
    """
    samples: List[Phase4Sample] = []
    data_root = ROOT_DIR / "Visualizar_Data" / "Data"

    normal_dir = data_root / "RetailS_train" / "pose" / "train"
    staged_dir = data_root / "RetailS_test_staged" / "pose" / "test"
    realworld_dir = data_root / "RetailS_test_realworld" / "pose" / "test"
    staged_mask_dir = data_root / "RetailS_test_staged" / "gt" / "test_frame_mask"
    realworld_mask_dir = data_root / "RetailS_test_realworld" / "gt" / "test_frame_mask"

    normal_files = sorted(normal_dir.glob("*.json"))
    staged_files = sorted(staged_dir.glob("*.json"))
    realworld_files = sorted(realworld_dir.glob("*.json"))

    if normal_file_limit is not None:
        normal_files = normal_files[:max(0, int(normal_file_limit))]
    if staged_file_limit is not None:
        staged_files = staged_files[:max(0, int(staged_file_limit))]
    if realworld_file_limit is not None:
        realworld_files = realworld_files[:max(0, int(realworld_file_limit))]

    for json_path in normal_files:
        try:
            coords, scores = _load_pose_clip(json_path)
        except Exception:
            continue
        samples.extend(
            _build_windows_from_clip(
                coords,
                scores,
                sequence_length=sequence_length,
                source=f"normal:{json_path.stem}",
                label=0,
                windows_per_clip=windows_per_clip,
            )
        )

    for json_path, mask_dir, split_name in (
        *[(path, staged_mask_dir, "staged") for path in staged_files],
        *[(path, realworld_mask_dir, "realworld") for path in realworld_files],
    ):
        try:
            coords, scores = _load_pose_clip(json_path)
            mask_path = mask_dir / f"{json_path.stem}.npy"
            mask = _load_mask_clip(mask_path, len(coords))
        except Exception:
            continue
        samples.extend(
            _build_windows_from_clip(
                coords,
                scores,
                sequence_length=sequence_length,
                source=f"{split_name}:{json_path.stem}",
                label=1,
                windows_per_clip=windows_per_clip,
                mask=mask,
            )
        )

    return samples