"""Pipeline modular do sistema anti-furto."""

from .config import AppConfig, ConfigError
from .spatial_normalizer import (
    SpatialNormalizer,
    NormalizationParams,
    NormalizedPose,
)
from .skeleton_visualizer import (
    SkeletonVisualizer,
    visualize_normalized_pose,
)
from .temporal_pose_filter import (
    TemporalPoseFilter,
    TemporalPoseFilterConfig,
)

__all__ = [
    "AppConfig",
    "ConfigError",
    "SpatialNormalizer",
    "NormalizationParams",
    "NormalizedPose",
    "SkeletonVisualizer",
    "visualize_normalized_pose",
    "TemporalPoseFilter",
    "TemporalPoseFilterConfig",
]
