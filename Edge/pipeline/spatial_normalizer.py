"""
2D Spatial Normalization for RTMO skeleton data (17 COCO keypoints).

Implements torso-normalized positions with root-relative translation and 
anatomical scale invariance using strictly vectorized numpy operations.

Key Features:
  - Root-relative translation: Pelvis (hip midpoint) centered at origin
  - Anatomical scale invariance: Normalize by torso length (neck-to-pelvis distance)
  - Config-driven confidence thresholds (no hardcoded values)
  - Vectorized numpy operations (zero Python loops over keypoints)
  - Type-safe with full type hints
  - Fast failure on invalid/low-confidence torso joints
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from Detecao.skeleton import (
    KEYPOINT_NAMES as COCO_KEYPOINT_NAMES,
    LEFT_HIP as PELVIS_LEFT_IDX,
    RIGHT_HIP as PELVIS_RIGHT_IDX,
    LEFT_SHOULDER as SHOULDER_LEFT_IDX,
    RIGHT_SHOULDER as SHOULDER_RIGHT_IDX
)



@dataclass
class NormalizationParams:
    """Configuration for spatial normalization."""
    
    torso_confidence_threshold: float = 0.5
    """Minimum confidence for torso joints (pelvis & shoulders) to be valid."""
    
    min_torso_length_px: float = 10.0
    """Minimum torso length (neck-to-pelvis) in pixels to avoid division by near-zero."""
    
    allow_invalid_torso: bool = False
    """If True, use previous valid normalization when torso confidence is low.
    If False, return all NaNs for invalid frames."""


@dataclass
class NormalizedPose:
    """Output of spatial normalization."""
    
    keypoints: np.ndarray
    """Normalized keypoints, shape (17, 2). NaN if frame invalid."""
    
    scores: np.ndarray
    """Confidence scores, shape (17,). Unchanged from input."""
    
    is_valid: bool
    """True if torso anchors passed confidence threshold."""
    
    pelvis: np.ndarray
    """Pelvis position (raw), shape (2,)."""
    
    neck: np.ndarray
    """Neck anchor position (raw), shape (2,)."""
    
    torso_length: float
    """Euclidean distance from pelvis to neck (before normalization)."""


class SpatialNormalizer:
    """
    Normalizes 17-keypoint COCO poses to torso-relative, scale-invariant space.
    
    Transformation:
      1. Translate: pelvis (hip midpoint) -> origin (0, 0)
      2. Scale: divide by torso length (neck-to-pelvis distance)
      
    Vectorized operations only—no Python loops over keypoints.
    """
    
    def __init__(self, config_source=None, params: Optional[NormalizationParams] = None):
        """
        Initialize normalizer.
        
        Args:
            config_source: AppConfig-like object or dict with spatial_normalization settings.
            params: NormalizationParams with confidence thresholds.
                    If provided, takes precedence over config_source.
        """
        if params is None and isinstance(config_source, NormalizationParams):
            params = config_source

        if params is not None:
            self.params = params
        else:
            spatial_cfg = {}
            if config_source is not None:
                getter = getattr(config_source, "data", None)
                if isinstance(getter, dict):
                    spatial_cfg = getter.get("spatial_normalization", {})
                elif isinstance(config_source, dict):
                    spatial_cfg = config_source.get("spatial_normalization", {})

            self.params = NormalizationParams(
                torso_confidence_threshold=spatial_cfg.get("torso_confidence_threshold", 0.5),
                min_torso_length_px=spatial_cfg.get("min_torso_length_px", 10.0),
                allow_invalid_torso=spatial_cfg.get("allow_invalid_torso", False),
            )
        self._prev_valid_pose: Optional[NormalizedPose] = None
    
    def normalize(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
    ) -> NormalizedPose:
        """
        Normalize a single frame's pose to torso-relative space.
        
        Args:
            keypoints: Shape (17, 2), raw pixel coordinates [x, y]
            scores: Shape (17,), confidence [0.0-1.0]
        
        Returns:
            NormalizedPose with normalized keypoints or NaNs if invalid
        
        Raises:
            ValueError: If shapes are invalid
            AssertionError: If dtypes are not as expected
        """
        # Validate shapes and dtypes
        if keypoints.shape != (17, 2):
            raise ValueError(f"Expected keypoints shape (17, 2), got {keypoints.shape}")
        if scores.shape != (17,):
            raise ValueError(f"Expected scores shape (17,), got {scores.shape}")
        
        # Ensure float32 for vectorized operations
        keypoints = np.asarray(keypoints, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        
        # ===== Compute Torso Anchors (Vectorized) =====
        
        # Pelvis: midpoint of left hip (11) and right hip (12)
        pelvis_left = keypoints[PELVIS_LEFT_IDX]      # [x, y]
        pelvis_right = keypoints[PELVIS_RIGHT_IDX]    # [x, y]
        pelvis = (pelvis_left + pelvis_right) * 0.5   # Vectorized average
        
        # Neck: midpoint of left shoulder (5) and right shoulder (6)
        shoulder_left = keypoints[SHOULDER_LEFT_IDX]  # [x, y]
        shoulder_right = keypoints[SHOULDER_RIGHT_IDX]  # [x, y]
        neck = (shoulder_left + shoulder_right) * 0.5  # Vectorized average
        
        # Torso confidence: min of the 4 torso joints
        pelvis_conf = np.minimum(
            scores[PELVIS_LEFT_IDX],
            scores[PELVIS_RIGHT_IDX]
        )
        neck_conf = np.minimum(
            scores[SHOULDER_LEFT_IDX],
            scores[SHOULDER_RIGHT_IDX]
        )
        torso_conf = np.minimum(pelvis_conf, neck_conf)
        
        # Check torso validity
        is_valid = bool(torso_conf >= self.params.torso_confidence_threshold)
        
        # Virtual Pelvis Fallback for waist-up/occluded hips
        if not is_valid and neck_conf >= self.params.torso_confidence_threshold:
            # Estimate pelvis as neck + [0, 1.2 * shoulder_width]
            shoulder_width = np.float32(np.linalg.norm(shoulder_left - shoulder_right))
            if shoulder_width >= self.params.min_torso_length_px:
                pelvis = neck + np.array([0.0, 1.2 * shoulder_width], dtype=np.float32)
                is_valid = True
        
        if not is_valid:
            # Return invalid frame with NaNs
            if self.params.allow_invalid_torso and self._prev_valid_pose is not None:
                return self._prev_valid_pose
            else:
                return NormalizedPose(
                    keypoints=np.full((17, 2), np.nan, dtype=np.float32),
                    scores=scores,
                    is_valid=False,
                    pelvis=pelvis,
                    neck=neck,
                    torso_length=0.0,
                )
        
        # ===== Compute Torso Length (Vectorized) =====
        torso_vec = neck - pelvis              # [dx, dy]
        torso_length = np.float32(np.linalg.norm(torso_vec))  # Euclidean distance (default de linalg é 64float->float32)
        
        if torso_length < self.params.min_torso_length_px:
            # Torso too small, return invalid
            if self.params.allow_invalid_torso and self._prev_valid_pose is not None:
                return self._prev_valid_pose
            else:
                return NormalizedPose(
                    keypoints=np.full((17, 2), np.nan, dtype=np.float32),
                    scores=scores,
                    is_valid=False,
                    pelvis=pelvis,
                    neck=neck,
                    torso_length=torso_length,
                )
        
        # ===== Normalize All Keypoints (Fully Vectorized) =====
        # Step 1: Translate by subtracting pelvis (broadcasting)
        centered_kpts = keypoints - pelvis[np.newaxis, :]  # (17, 2) - (1, 2)
        
        # Step 2: Scale by dividing by torso length
        normalized_kpts = centered_kpts / torso_length     # (17, 2) / scalar
        
        # Create result
        result = NormalizedPose(
            keypoints=normalized_kpts.astype(np.float32),
            scores=scores,
            is_valid=True,
            pelvis=pelvis.astype(np.float32),
            neck=neck.astype(np.float32),
            torso_length=float(torso_length),
        )
        
        # Cache valid pose for fallback
        self._prev_valid_pose = result
        
        return result