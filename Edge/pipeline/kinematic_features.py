from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray
from Detecao.skeleton import DEFAULT_LIMBS



@dataclass(frozen=True)
class KinematicConfig:
    """Configuration for kinematic feature extraction."""
    eps: float = 1e-8


class KinematicFeatureExtractor:
    """Vectorized kinematic feature extractor for COCO-17 poses.

    Input: coords array shape (B, T, 17, 2) — torso-normalized X,Y coordinates
    Output: features array shape (B, T, D) ready for PyTorch LSTM

    Features produced (concatenated):
      - per-keypoint velocity (dx, dy) flattened -> 17*2 = 34
      - limb orientation unit vectors (cos, sin) for selected limbs -> 2 * L

    All operations are vectorized; no Python loops over batch/frame/keypoint.
    """

    # COCO keypoint indices
    # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
    # 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
    # 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
    # 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

    DEFAULT_LIMBS: List[Tuple[int, int]] = DEFAULT_LIMBS


    def __init__(self, limbs: List[Tuple[int, int]] = None, config: KinematicConfig = KinematicConfig()) -> None:
        self.limbs = limbs if limbs is not None else self.DEFAULT_LIMBS
        self.config = config

    def feature_dim(self) -> int:
        # velocities: 17*2, limb dirs: 2 * num_limbs, distances: 10
        return 17 * 2 + 2 * len(self.limbs) + 10

    def transform(self, coords: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute kinematic features from input coordinates.

        Args:
            coords: np.ndarray shape (B, T, 17, 2)

        Returns:
            features: np.ndarray shape (B, T, feature_dim)
        """
        if coords.ndim != 4 or coords.shape[2] != 17 or coords.shape[3] != 2:
            raise ValueError("coords must have shape (B, T, 17, 2)")

        # Ensure float32 for downstream PyTorch consumption
        coords = coords.astype(np.float32, copy=False)

        B, T, K, C = coords.shape  # K should be 17, C==2

        # Mask for missing keypoints: True where any coord is NaN
        missing_mask = np.isnan(coords).any(axis=-1)  # (B, T, K)

        # Replace NaNs with zeros for safe arithmetic
        coords_clean = np.where(np.isnan(coords), 0.0, coords)

        # Temporal velocity: first difference along time axis, pad first frame with zeros
        # diffs shape (B, T-1, K, 2)
        diffs = coords_clean[:, 1:, :, :] - coords_clean[:, :-1, :, :]
        # pad zeros for first frame
        zeros = np.zeros((B, 1, K, C), dtype=np.float32)
        velocities = np.concatenate([zeros, diffs], axis=1)  # (B, T, K, 2)

        # Mask velocities where either current or previous keypoint was missing
        missing_prev = np.concatenate([np.zeros((B, 1, K), dtype=bool), missing_mask[:, :-1, :]], axis=1)
        invalid_velocity = missing_mask | missing_prev  # (B, T, K)
        velocities = np.where(invalid_velocity[..., None], 0.0, velocities)

        # Flatten velocities to (B, T, 34)
        vel_flat = velocities.reshape(B, T, K * C)

        # Limb orientation vectors: from parent -> child (child - parent)
        # Build arrays for indexing
        parents = np.array([p for (p, c) in self.limbs], dtype=np.int64)
        children = np.array([c for (p, c) in self.limbs], dtype=np.int64)

        # Gather coordinates: shape (B, T, L, 2)
        parent_coords = coords_clean[:, :, parents, :]  # (B, T, L, 2)
        child_coords = coords_clean[:, :, children, :]  # (B, T, L, 2)

        limb_vecs = child_coords - parent_coords  # (B, T, L, 2)

        # Limb validity mask: True if either endpoint missing
        limb_missing = (missing_mask[:, :, parents] | missing_mask[:, :, children])  # (B, T, L)

        # Compute norms
        dx = limb_vecs[..., 0]
        dy = limb_vecs[..., 1]
        norm = np.sqrt(dx * dx + dy * dy)

        # Avoid division by zero without triggering a runtime warning.
        eps = float(self.config.eps)
        inv_norm = np.zeros_like(norm, dtype=np.float32)
        np.divide(1.0, norm, out=inv_norm, where=norm > eps)

        ux = dx * inv_norm
        uy = dy * inv_norm

        # Where limb is missing or zero-length, set unit vectors to zero
        ux = np.where(limb_missing, 0.0, ux)
        uy = np.where(limb_missing, 0.0, uy)

        # Stack unit vectors as (cos, sin) per limb
        limb_dirs = np.stack([ux, uy], axis=-1)  # (B, T, L, 2)

        # Flatten limb dirs to (B, T, 2*L)
        limb_flat = limb_dirs.reshape(B, T, -1)

        # Gather key coordinates for relative distances (B, T, 2)
        lwrist = coords_clean[:, :, 9, :]
        rwrist = coords_clean[:, :, 10, :]
        lhip = coords_clean[:, :, 11, :]
        rhip = coords_clean[:, :, 12, :]
        lshoulder = coords_clean[:, :, 5, :]
        rshoulder = coords_clean[:, :, 6, :]
        nose = coords_clean[:, :, 0, :]

        # Compute Euclidean distances
        d_lw_lh = np.linalg.norm(lwrist - lhip, axis=-1, keepdims=True)
        d_lw_rh = np.linalg.norm(lwrist - rhip, axis=-1, keepdims=True)
        d_lw_ls = np.linalg.norm(lwrist - lshoulder, axis=-1, keepdims=True)
        d_lw_rs = np.linalg.norm(lwrist - rshoulder, axis=-1, keepdims=True)
        d_lw_nose = np.linalg.norm(lwrist - nose, axis=-1, keepdims=True)

        d_rw_rh = np.linalg.norm(rwrist - rhip, axis=-1, keepdims=True)
        d_rw_lh = np.linalg.norm(rwrist - lhip, axis=-1, keepdims=True)
        d_rw_rs = np.linalg.norm(rwrist - rshoulder, axis=-1, keepdims=True)
        d_rw_ls = np.linalg.norm(rwrist - lshoulder, axis=-1, keepdims=True)
        d_rw_nose = np.linalg.norm(rwrist - nose, axis=-1, keepdims=True)

        # Concatenate distance features: (B, T, 10)
        distances = np.concatenate([
            d_lw_lh, d_lw_rh, d_lw_ls, d_lw_rs, d_lw_nose,
            d_rw_rh, d_rw_lh, d_rw_rs, d_rw_ls, d_rw_nose
        ], axis=-1)

        # Handle missing keypoint masks for distances (set distance to 0.0 if either endpoint is missing)
        m_lw = missing_mask[:, :, 9]
        m_rw = missing_mask[:, :, 10]
        m_lh = missing_mask[:, :, 11]
        m_rh = missing_mask[:, :, 12]
        m_ls = missing_mask[:, :, 5]
        m_rs = missing_mask[:, :, 6]
        m_nose = missing_mask[:, :, 0]

        mask_lw_lh = (m_lw | m_lh)[..., None]
        mask_lw_rh = (m_lw | m_rh)[..., None]
        mask_lw_ls = (m_lw | m_ls)[..., None]
        mask_lw_rs = (m_lw | m_rs)[..., None]
        mask_lw_nose = (m_lw | m_nose)[..., None]

        mask_rw_rh = (m_rw | m_rh)[..., None]
        mask_rw_lh = (m_rw | m_lh)[..., None]
        mask_rw_rs = (m_rw | m_rs)[..., None]
        mask_rw_ls = (m_rw | m_ls)[..., None]
        mask_rw_nose = (m_rw | m_nose)[..., None]

        dist_masks = np.concatenate([
            mask_lw_lh, mask_lw_rh, mask_lw_ls, mask_lw_rs, mask_lw_nose,
            mask_rw_rh, mask_rw_lh, mask_rw_rs, mask_rw_ls, mask_rw_nose
        ], axis=-1)

        distances = np.where(dist_masks, 0.0, distances)

        # Concatenate velocity, limb orientation, and distance features
        features = np.concatenate([vel_flat, limb_flat, distances], axis=-1)

        # Final safety: replace any NaNs or infs with zeros
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        return features
