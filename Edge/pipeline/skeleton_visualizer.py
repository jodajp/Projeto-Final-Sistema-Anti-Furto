"""
Debug Visualization for Spatially Normalized Skeleton Data.

Renders normalized poses to a fixed-size grid (default 500x500) with
skeleton connections, keypoint labels, and confidence information.

Designed for real-time sanity checking of translation and scale invariance.
"""

from typing import Optional, Tuple
import numpy as np
import cv2


from Detecao.skeleton import (
    SKELETON_CONNECTIONS,
    KEYPOINT_NAMES,
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_SHOULDER,
    RIGHT_SHOULDER
)


# Color scheme: BGR (OpenCV format)
COLOR_SKELETON_LINE = (0, 255, 255)    # Cyan
COLOR_KEYPOINT = (0, 255, 0)           # Green
COLOR_PELVIS = (255, 0, 0)             # Blue (special highlight)
COLOR_NECK = (0, 0, 255)               # Red (special highlight)
COLOR_TEXT = (200, 200, 200)           # Light gray
COLOR_INVALID = (100, 100, 100)        # Dark gray
COLOR_BACKGROUND = (0, 0, 0)           # Black


class SkeletonVisualizer:
    """
    Renders normalized skeleton poses to a 2D canvas.
    
    Features:
      - Fixed grid size (configurable, default 500x500)
      - Automatic scaling from normalized coordinates to pixel space
      - Skeleton line connections with confidence-based transparency
      - Keypoint labels and confidence values
      - Special highlighting for torso anchors (pelvis, neck)
      - Handles invalid frames gracefully
    """
    
    def __init__(
        self,
        canvas_size: int = 500,
        margin_px: int = 20,
        point_radius: int = 4,
        line_thickness: int = 2,
        show_labels: bool = True,
        show_confidence: bool = True,
    ):
        """
        Initialize visualizer.
        
        Args:
            canvas_size: Width/height of output image in pixels
            margin_px: Margin around skeleton bounds
            point_radius: Radius of keypoint circles
            line_thickness: Thickness of skeleton connection lines
            show_labels: Display keypoint names
            show_confidence: Display confidence scores
        """
        self.canvas_size = canvas_size
        self.margin_px = margin_px
        self.point_radius = point_radius
        self.line_thickness = line_thickness
        self.show_labels = show_labels
        self.show_confidence = show_confidence
    
    def render(
        self,
        normalized_keypoints: np.ndarray,
        scores: Optional[np.ndarray] = None,
        title: str = "Normalized Skeleton",
    ) -> np.ndarray:
        """
        Render normalized skeleton to canvas.
        
        Args:
            normalized_keypoints: Shape (17, 2), normalized coordinates
            scores: Shape (17,), optional confidence scores
            title: Text to display at top of canvas
        
        Returns:
            Canvas image, shape (canvas_size, canvas_size, 3), BGR uint8
        
        Raises:
            ValueError: If keypoints shape is invalid
        """
        if normalized_keypoints.shape != (17, 2):
            raise ValueError(
                f"Expected keypoints shape (17, 2), got {normalized_keypoints.shape}"
            )
        
        if scores is None:
            scores = np.ones(17, dtype=np.float32)
        else:
            scores = np.asarray(scores, dtype=np.float32)
            if scores.shape != (17,):
                raise ValueError(f"Expected scores shape (17,), got {scores.shape}")
        
        # Create black canvas
        canvas = np.zeros(
            (self.canvas_size, self.canvas_size, 3),
            dtype=np.uint8
        )
        
        # Add title
        if title:
            cv2.putText(
                canvas, title,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                COLOR_TEXT,
                1,
            )
        
        # ===== Scale normalized coordinates to pixel space =====
        kpts_px = self._normalize_to_pixels(normalized_keypoints)
        
        # ===== Draw skeleton connections =====
        for idx_from, idx_to in SKELETON_CONNECTIONS:
            pt_from = kpts_px[idx_from]
            pt_to = kpts_px[idx_to]
            
            # Check if both points are valid (not NaN)
            if np.isnan(pt_from).any() or np.isnan(pt_to).any():
                continue
            
            # Confidence-weighted line thickness (fade weak connections)
            conf_avg = (scores[idx_from] + scores[idx_to]) / 2.0
            line_thickness = max(1, int(self.line_thickness * conf_avg))
            
            pt_from_int = tuple(map(int, pt_from))
            pt_to_int = tuple(map(int, pt_to))
            
            cv2.line(
                canvas,
                pt_from_int,
                pt_to_int,
                COLOR_SKELETON_LINE,
                line_thickness,
                lineType=cv2.LINE_AA,
            )
        
        # ===== Draw keypoints =====
        for idx in range(17):
            pt = kpts_px[idx]
            
            if np.isnan(pt).any():
                continue
            
            pt_int = tuple(map(int, pt))
            conf = scores[idx]
            
            # Special colors for torso anchors
            if idx in [LEFT_HIP, RIGHT_HIP]:  # Pelvis
                color = COLOR_PELVIS
                radius = self.point_radius + 1
            elif idx in [LEFT_SHOULDER, RIGHT_SHOULDER]:  # Shoulders (neck anchor)
                color = COLOR_NECK
                radius = self.point_radius + 1

            else:
                color = COLOR_KEYPOINT
                radius = self.point_radius
            
            # Draw circle
            cv2.circle(
                canvas,
                pt_int,
                radius,
                color,
                -1,  # filled
                lineType=cv2.LINE_AA,
            )
            
            # Draw confidence ring (outer circle)
            ring_radius = int(radius * conf)
            if ring_radius > 0:
                cv2.circle(
                    canvas,
                    pt_int,
                    ring_radius,
                    (200, 200, 200),
                    1,
                    lineType=cv2.LINE_AA,
                )
            
            # Draw label if enabled
            if self.show_labels:
                label = KEYPOINT_NAMES[idx]
                if self.show_confidence:
                    label += f" {conf:.2f}"
                
                cv2.putText(
                    canvas,
                    label,
                    (pt_int[0] + 8, pt_int[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    COLOR_TEXT,
                    1,
                )
        
        # Draw origin marker (pelvis should be near center for normalized poses)
        origin_px = self._normalize_to_pixels(np.array([[0.0, 0.0]]))[0]
        if not np.isnan(origin_px).any():
            origin_px_int = tuple(map(int, origin_px))
            cv2.drawMarker(
                canvas,
                origin_px_int,
                (255, 0, 255),  # Magenta
                markerType=cv2.MARKER_CROSS,
                markerSize=15,
                thickness=2,
            )
        
        return canvas
    
    def _normalize_to_pixels(self, norm_coords: np.ndarray) -> np.ndarray:
        """
        Convert normalized coordinates to pixel space.
        
        Normalized coords are typically in range [-1, 1] after scale normalization.
        We scale them to fit the canvas with a margin.
        
        Args:
            norm_coords: Shape (N, 2), normalized coordinates
        
        Returns:
            Pixel coordinates, shape (N, 2), clamped to canvas bounds
        """
        valid_mask = ~np.isnan(norm_coords).any(axis=1)
        if not valid_mask.any():
            return np.full_like(norm_coords, np.nan)
        
        valid_coords = norm_coords[valid_mask]
        x_min, x_max = valid_coords[:, 0].min(), valid_coords[:, 0].max()
        y_min, y_max = valid_coords[:, 1].min(), valid_coords[:, 1].max()
        
        if x_min == x_max:
            x_min, x_max = x_min - 0.5, x_max + 0.5
        if y_min == y_max:
            y_min, y_max = y_min - 0.5, y_max + 0.5
        
        scale = (self.canvas_size - 2 * self.margin_px) / max(x_max - x_min, y_max - y_min)
        
        px_coords = np.empty_like(norm_coords)
        px_coords[:, 0] = (norm_coords[:, 0] - x_min) * scale + self.margin_px
        px_coords[:, 1] = (norm_coords[:, 1] - y_min) * scale + self.margin_px
        
        px_coords = np.clip(px_coords, 0, self.canvas_size - 1)
        px_coords[~valid_mask] = np.nan
        return px_coords
