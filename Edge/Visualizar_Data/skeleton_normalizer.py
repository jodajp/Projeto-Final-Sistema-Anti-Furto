#!/usr/bin/env python3
"""
Skeleton Normalization & Centering Module
Provides utilities for normalizing and centering skeleton keypoints for video generation.

Features:
- Center skeleton around pelvis (center of mass)
- Scale to fit frame while maintaining aspect ratio
- 90-degree rotation for proper orientation (feet downward)
- Vectorized NumPy operations (no Python loops)
- Config-driven parameters
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class SkeletonNormConfig:
    """Configuration for skeleton normalization"""
    confidence_threshold: float = 0.3
    canvas_width: int = 800
    canvas_height: int = 1400
    padding_ratio: float = 0.15  # 15% padding around skeleton
    apply_rotation_90deg: bool = True  # Rotate so feet point down
    center_on_pelvis: bool = True  # Center around hip midpoint


class SkeletonNormalizer:
    """
    Normalize and center skeleton keypoints for video rendering.
    
    Provides:
    1. Confidence-based filtering
    2. Pelvis-centered positioning
    3. Scale-invariant sizing
    4. 90-degree rotation for proper orientation
    5. Canvas-space pixel conversion
    """
    
    # COCO keypoint indices
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
    
    def __init__(self, config: SkeletonNormConfig):
        self.config = config
    
    def normalize_and_center(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Normalize skeleton by:
        1. Filtering by confidence threshold
        2. Computing pelvis center
        3. Centering around pelvis
        4. Scaling to fit canvas with padding
        5. Applying 90° rotation
        6. Converting to pixel coordinates
        
        Args:
            keypoints: Shape (17, 3) with [x, y, confidence]
        
        Returns:
            Pixel coordinates shape (17, 3) ready for drawing
        """
        kp_out = keypoints.copy().astype(np.float32)
        
        # Step 1: Get valid keypoints based on confidence
        valid_mask = keypoints[:, 2] >= self.config.confidence_threshold
        valid_kp = keypoints[valid_mask]
        
        if len(valid_kp) == 0:
            # No valid keypoints, return original with adjusted coordinates
            return self._to_pixel_space(kp_out)
        
        # Step 2: Calculate pelvis center (midpoint of hips)
        if (valid_mask[self.LEFT_HIP] and valid_mask[self.RIGHT_HIP]):
            pelvis_x = (keypoints[self.LEFT_HIP, 0] + keypoints[self.RIGHT_HIP, 0]) / 2.0
            pelvis_y = (keypoints[self.LEFT_HIP, 1] + keypoints[self.RIGHT_HIP, 1]) / 2.0
        else:
            # Fallback to centroid of all valid points
            pelvis_x = np.mean(valid_kp[:, 0])
            pelvis_y = np.mean(valid_kp[:, 1])
        
        # Step 3: Center all keypoints around pelvis
        centered_kp = keypoints.copy()
        centered_kp[:, 0] -= pelvis_x
        centered_kp[:, 1] -= pelvis_y
        
        # Step 4: Calculate bounding box of valid centered keypoints
        centered_valid = centered_kp[valid_mask]
        x_min, x_max = centered_valid[:, 0].min(), centered_valid[:, 0].max()
        y_min, y_max = centered_valid[:, 1].min(), centered_valid[:, 1].max()
        
        width = x_max - x_min
        height = y_max - y_min
        
        # Add padding
        padding_x = width * self.config.padding_ratio
        padding_y = height * self.config.padding_ratio
        
        width_padded = width + 2 * padding_x
        height_padded = height + 2 * padding_y
        
        # Calculate scale to fit in canvas
        scale_x = self.config.canvas_width / width_padded if width_padded > 0 else 1.0
        scale_y = self.config.canvas_height / height_padded if height_padded > 0 else 1.0
        scale = min(scale_x, scale_y, 2.0)  # Cap scaling to prevent overgrowth
        
        # Step 5: Apply scaling
        scaled_kp = centered_kp.copy()
        scaled_kp[:, 0] *= scale
        scaled_kp[:, 1] *= scale
        
        # Step 6: Translate to canvas center
        scaled_valid = scaled_kp[valid_mask]
        scaled_x_min = scaled_valid[:, 0].min()
        scaled_y_min = scaled_valid[:, 1].min()
        
        translate_x = self.config.padding_ratio * self.config.canvas_width / 2 - scaled_x_min
        translate_y = self.config.padding_ratio * self.config.canvas_height / 2 - scaled_y_min
        
        translated_kp = scaled_kp.copy()
        translated_kp[:, 0] += translate_x + self.config.canvas_width / 2
        translated_kp[:, 1] += translate_y + self.config.canvas_height / 2
        
        # Step 7: Apply 90-degree rotation if enabled
        if self.config.apply_rotation_90deg:
            rotated_kp = self._rotate_90_clockwise(translated_kp)
        else:
            rotated_kp = translated_kp
        
        # Step 8: Flip Y for OpenCV coordinate system
        rotated_kp[:, 1] = self.config.canvas_height - rotated_kp[:, 1]
        
        # Clip to canvas bounds
        rotated_kp[:, 0] = np.clip(rotated_kp[:, 0], 0, self.config.canvas_width)
        rotated_kp[:, 1] = np.clip(rotated_kp[:, 1], 0, self.config.canvas_height)
        
        return rotated_kp
    
    def _rotate_90_clockwise(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Rotate keypoints 90 degrees clockwise.
        Formula: (x', y') = (canvas_height - y, x)
        """
        kp_rot = keypoints.copy()
        x = keypoints[:, 0]
        y = keypoints[:, 1]
        
        # Center rotation point at canvas center
        cx = self.config.canvas_width / 2
        cy = self.config.canvas_height / 2
        
        # Translate to origin, rotate, translate back
        x_centered = x - cx
        y_centered = y - cy
        
        x_rot = -y_centered + cx
        y_rot = x_centered + cy
        
        kp_rot[:, 0] = x_rot
        kp_rot[:, 1] = y_rot
        
        return kp_rot
    
    def _to_pixel_space(self, keypoints: np.ndarray) -> np.ndarray:
        """Convert keypoints to pixel space (just Y-flip for OpenCV)"""
        kp_px = keypoints.copy()
        kp_px[:, 1] = self.config.canvas_height - kp_px[:, 1]
        kp_px[:, 0] = np.clip(kp_px[:, 0], 0, self.config.canvas_width)
        kp_px[:, 1] = np.clip(kp_px[:, 1], 0, self.config.canvas_height)
        return kp_px
    
    @staticmethod
    def compute_skeleton_bounds(keypoints: np.ndarray, 
                               confidence_threshold: float = 0.3) -> Tuple[float, float, float, float]:
        """
        Compute bounding box of skeleton (vectorized).
        Returns: (x_min, y_min, x_max, y_max)
        """
        valid_mask = keypoints[:, 2] >= confidence_threshold
        valid_kp = keypoints[valid_mask]
        
        if len(valid_kp) == 0:
            return 0.0, 0.0, 0.0, 0.0
        
        x_min = float(valid_kp[:, 0].min())
        x_max = float(valid_kp[:, 0].max())
        y_min = float(valid_kp[:, 1].min())
        y_max = float(valid_kp[:, 1].max())
        
        return x_min, y_min, x_max, y_max
    
    @staticmethod
    def compute_skeleton_scale(keypoints: np.ndarray,
                              target_width: int,
                              target_height: int,
                              confidence_threshold: float = 0.3) -> Tuple[float, float, float]:
        """
        Compute scale factor to fit skeleton in target canvas.
        Returns: (scale, scale_x, scale_y)
        """
        x_min, y_min, x_max, y_max = SkeletonNormalizer.compute_skeleton_bounds(
            keypoints, confidence_threshold
        )
        
        width = x_max - x_min
        height = y_max - y_min
        
        if width == 0 or height == 0:
            return 1.0, 1.0, 1.0
        
        scale_x = target_width / width
        scale_y = target_height / height
        scale = min(scale_x, scale_y)
        
        return scale, scale_x, scale_y
