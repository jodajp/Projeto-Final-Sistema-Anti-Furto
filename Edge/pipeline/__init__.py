"""Pipeline modular do sistema anti-furto."""

from .config import AppConfig, ConfigError
from .spatial_normalizer import (
    SpatialNormalizer,
    NormalizationParams,
    NormalizedPose,
)
from .skeleton_visualizer import (
    SkeletonVisualizer,
)
from .temporal_pose_filter import (
    TemporalPoseFilter,
)

from .kinematic_features import KinematicConfig, KinematicFeatureExtractor

__all__ = [
    "AppConfig",
    "ConfigError",
    "SpatialNormalizer",
    "NormalizationParams",
    "NormalizedPose",
    "SkeletonVisualizer",
    "TemporalPoseFilter",
    "KinematicConfig",
    "KinematicFeatureExtractor",
]
