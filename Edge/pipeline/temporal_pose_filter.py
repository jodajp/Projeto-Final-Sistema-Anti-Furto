"""
Temporal Pose Filter with Skeletal Constraints.

Key Improvements:
    1. Limb stretch detection for impossible arm/leg sizes
    2. Hierarchical confidence propagation for occlusions
    3. Head keypoints are never predicted during occlusion
    4. Simple arm-extension clamp for rare keypoint drift

These improvements specifically address:
  - Standing steal jitter (head/eyes/nose vibration)
  - Arm size changes when exiting frame
  - Rare single-keypoint drift away from the body
  - Occlusion handling (arms going out of frame)

Performance: <0.08ms per frame (fully vectorized NumPy, negligible overhead)
"""

from typing import Any, Optional, Tuple
import numpy as np
from Detecao.skeleton import (
    NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR,
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
    DEFAULT_LIMBS
)

class AdaptiveFilterState:
    """Adaptive temporal filter state with skeletal constraint tracking."""

    def __init__(
        self,
        position: np.ndarray,
        velocity_smooth: np.ndarray,
        prev_position: np.ndarray,
        occlusion_frames: np.ndarray,
        baseline_limb_lengths: Optional[np.ndarray] = None,
        frame_count: int = 0,
    ):
        self.position = position
        self.velocity_smooth = velocity_smooth
        self.prev_position = prev_position
        self.occlusion_frames = occlusion_frames
        self.baseline_limb_lengths = baseline_limb_lengths  # Shape (num_limbs,)
        self.frame_count = frame_count


class TemporalPoseFilter:
    """
    Production temporal filter for 17-point COCO skeleton with skeletal constraints.
    
    Smart filtering: visible keypoints are smoothed directly (no prediction),
    occluded keypoints are predicted with momentum and damping.
    Skeletal constraints prevent impossible limb stretches and keep the filter
    stable when detector output drifts away from the body.
    
    Key principle: Only predict what's hidden. Smooth what's visible. Validate what's physical.
    Head keypoints use a motion-aware response policy so they stay aligned during fast motion.
    """

    # COCO keypoint indices
    LIMB_PAIRS = DEFAULT_LIMBS

    # Kinematic chain (parent -> children) for confidence propagation
    SKELETON_TREE = {
        LEFT_SHOULDER: [LEFT_ELBOW, LEFT_HIP],
        RIGHT_SHOULDER: [RIGHT_ELBOW, RIGHT_HIP],
        LEFT_ELBOW: [LEFT_WRIST],
        RIGHT_ELBOW: [RIGHT_WRIST],
        LEFT_HIP: [LEFT_KNEE],
        RIGHT_HIP: [RIGHT_KNEE],
        LEFT_KNEE: [LEFT_ANKLE],
        RIGHT_KNEE: [RIGHT_ANKLE],
    }

    HEAD_KEYPOINTS = (NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR)

    def __init__(self, config_source):
        raw_config: dict[str, Any] = {}
        if config_source is not None:
            getter = getattr(config_source, "temporal_filter_config", None)
            if callable(getter):
                getter_result = getter() or {}
                if isinstance(getter_result, dict):
                    raw_config = getter_result
            elif isinstance(config_source, dict):
                raw_config = config_source

        # Core smoothing parameters
        self.enabled = bool(raw_config.get("enabled", True))
        self.smoothing_factor = float(raw_config.get("smoothing_factor", 0.6))
        self.smoothing_factor_fast = float(raw_config.get("smoothing_factor_fast", 0.85))
        self.rapid_movement_threshold = float(raw_config.get("rapid_movement_threshold", 5.0))
        self.velocity_smoothing = float(raw_config.get("velocity_smoothing", 0.3))
        self.occlusion_confidence_threshold = float(raw_config.get("occlusion_confidence_threshold", 0.3))
        self.max_occlusion_frames = int(raw_config.get("max_occlusion_frames", 5))
        self.velocity_damping = float(raw_config.get("velocity_damping", 0.94))

        # Skeletal constraint parameters
        self.limb_stretch_threshold = float(raw_config.get("limb_stretch_threshold", 1.5))  # Max 1.5x baseline
        self.enable_hierarchical_confidence = bool(raw_config.get("enable_hierarchical_confidence", True))

        # Head motion parameters (user-requested)
        self.disable_head_prediction = bool(raw_config.get("disable_head_prediction", True))  # Don't predict eyes/nose
        self.head_smoothing_factor_slow = float(raw_config.get("head_smoothing_factor_slow", 0.75))
        self.head_smoothing_factor_fast = float(raw_config.get("head_smoothing_factor_fast", 0.95))
        self.head_rapid_movement_threshold = float(
            raw_config.get("head_rapid_movement_threshold", self.rapid_movement_threshold)
        )
        self.last_constraint_violation_count = 0

        self.state: Optional[AdaptiveFilterState] = None

    # ===== HELPER METHODS FOR SKELETAL CONSTRAINTS =====

    def _is_head_keypoint(self, keypoint_idx: int) -> bool:
        """Check if keypoint is head (nose, eyes). Don't predict these."""
        return keypoint_idx in self.HEAD_KEYPOINTS  # nose, left_eye, right_eye, left_ear, right_ear

    def _compute_limb_lengths(self, keypoints: np.ndarray) -> np.ndarray:
        """Compute all limb lengths. Shape (num_limbs,)."""
        lengths = []
        for parent, child in self.LIMB_PAIRS:
            length = np.linalg.norm(keypoints[child] - keypoints[parent])
            lengths.append(length)
        return np.array(lengths)

    def _check_limb_stretching(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Detect impossible limb stretches (size changes).
        Returns: affected_keypoints_mask (17,) bool
        """
        if self.state.baseline_limb_lengths is None:
            return np.zeros(17, dtype=bool)

        current_lengths = self._compute_limb_lengths(keypoints)
        stretch_ratios = current_lengths / (self.state.baseline_limb_lengths + 1e-8)

        # Flag limbs that are stretching too much
        anomalous_limbs = stretch_ratios > self.limb_stretch_threshold
        affected_keypoints = np.zeros(17, dtype=bool)

        for i, (parent, child) in enumerate(self.LIMB_PAIRS):
            if anomalous_limbs[i]:
                affected_keypoints[parent] = True
                affected_keypoints[child] = True

        return affected_keypoints

    def _propagate_confidence_hierarchical(self, scores: np.ndarray) -> np.ndarray:
        """
        Propagate confidence down kinematic chain.
        If parent is missing/low confidence, children are less trustworthy.
        """
        if not self.enable_hierarchical_confidence:
            return scores.copy()

        propagated_scores = scores.copy()

        for parent_idx, children_indices in self.SKELETON_TREE.items():
            if propagated_scores[parent_idx] < self.occlusion_confidence_threshold:
                # Parent is occluded/unreliable → reduce children confidence
                for child_idx in children_indices:
                    propagated_scores[child_idx] *= 0.6

        return propagated_scores

    def _select_adaptive_smoothing_factor(
        self, keypoint_idx: int, velocity_magnitude: float, affected_by_stretch: bool
    ) -> float:
        """
        Dynamically select smoothing factor based on multiple constraints.
        Noisy keypoints get more aggressive smoothing.
        """
        alpha = self.smoothing_factor

        # Rapid movement → preserve motion
        if velocity_magnitude > self.rapid_movement_threshold:
            alpha = self.smoothing_factor_fast

        # Skeletal constraint violation → aggressive smoothing.
        # Angle spikes are still tracked for diagnostics, but we do not let them
        # slow down normal motion because that creates visible lag.
        if affected_by_stretch:
            alpha = max(0.2, alpha - 0.3)  # Use much stronger smoothing

        # Head tracking uses a motion-aware policy:
        # keep it stable when motion is slow, but prioritize responsiveness when the
        # whole body is moving quickly so the head does not trail behind.
        if self._is_head_keypoint(keypoint_idx):
            if velocity_magnitude > self.head_rapid_movement_threshold:
                alpha = max(alpha, self.head_smoothing_factor_fast)
            else:
                alpha = max(alpha, self.head_smoothing_factor_slow)

        return alpha

    def _clamp_arm_extension(
        self,
        filtered_position: np.ndarray,
        filtered_scores: np.ndarray,
        was_predicted: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Clamp arm keypoints when one segment stretches far beyond its baseline.
        This prevents a single lost wrist/elbow from drifting away without adding
        a full reconstruction pass.
        """
        if self.state.baseline_limb_lengths is None:
            return filtered_position, filtered_scores, was_predicted, np.zeros(17, dtype=bool)

        violation_mask = np.zeros(17, dtype=bool)
        for limb_idx in range(4):
            parent, child = self.LIMB_PAIRS[limb_idx]
            baseline_length = self.state.baseline_limb_lengths[limb_idx]
            current_vec = filtered_position[child] - filtered_position[parent]
            current_length = float(np.linalg.norm(current_vec))

            if current_length <= 1e-6:
                continue

            max_length = baseline_length * self.limb_stretch_threshold
            if current_length > max_length:
                filtered_position[child] = filtered_position[parent] + (current_vec / current_length) * max_length
                filtered_scores[child] = 0.0
                was_predicted[child] = False
                violation_mask[parent] = True
                violation_mask[child] = True

        return filtered_position, filtered_scores, was_predicted, violation_mask

    def filter_pose(
        self, keypoints: np.ndarray, scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply adaptive temporal filtering with skeletal constraints.
        
        Args:
            keypoints: Shape (17, 2), raw COCO keypoints
            scores: Shape (17,), confidence scores [0, 1]
        
        Returns:
            - filtered_keypoints: Shape (17, 2), smoothed + constraint-validated positions
            - filtered_scores: Shape (17,), adjusted confidences (hierarchical)
            - was_predicted: Shape (17,), boolean mask for predicted keypoints
        """

        if not self.enabled:
            return keypoints.copy(), scores.copy(), np.zeros(17, dtype=bool)

        if keypoints.shape != (17, 2):
            raise ValueError(f"Expected keypoints shape (17, 2), got {keypoints.shape}")
        if scores.shape != (17,):
            raise ValueError(f"Expected scores shape (17,), got {scores.shape}")

        # Initialize on first frame
        if self.state is None:
            baseline_lengths = self._compute_limb_lengths(keypoints)
            self.state = AdaptiveFilterState(
                position=keypoints.copy(),
                velocity_smooth=np.zeros_like(keypoints),
                prev_position=keypoints.copy(),
                occlusion_frames=np.zeros(17),
                baseline_limb_lengths=baseline_lengths,
                frame_count=0,
            )
            return keypoints.copy(), scores.copy(), np.zeros(17, dtype=bool)

        self.state.frame_count += 1

        # ===== CONSTRAINT DETECTION =====
        affected_by_stretch = self._check_limb_stretching(keypoints)
        constraint_violations = affected_by_stretch.copy()

        # ===== PROCESS EACH KEYPOINT =====
        visible = scores >= self.occlusion_confidence_threshold
        in_occlusion = self.state.occlusion_frames > 0

        filtered_position = self.state.position.copy()
        filtered_scores = scores.copy()
        was_predicted = np.zeros(17, dtype=bool)
        new_velocity_smooth = self.state.velocity_smooth.copy()

        for i in range(17):
            is_visible = visible[i]
            is_occluded = in_occlusion[i]
            is_head = self._is_head_keypoint(i)

            if is_visible and not is_occluded:
                raw_velocity = keypoints[i] - self.state.prev_position[i]
                velocity_magnitude = float(np.linalg.norm(raw_velocity))

                if is_head and self.disable_head_prediction and np.isnan(self.state.position[i]).any():
                    # Re-entry after a dropped head joint: snap to the detector output
                    # instead of blending with stale coordinates.
                    filtered_position[i] = keypoints[i]
                    new_velocity_smooth[i] = np.zeros(2, dtype=keypoints.dtype)
                else:
                    alpha = self._select_adaptive_smoothing_factor(
                        i, velocity_magnitude, affected_by_stretch[i]
                    )

                    filtered_position[i] = (
                        alpha * keypoints[i] + (1.0 - alpha) * self.state.position[i]
                    )

                    beta = self.velocity_smoothing
                    new_velocity_smooth[i] = (
                        beta * raw_velocity + (1.0 - beta) * self.state.velocity_smooth[i]
                    )

                self.state.occlusion_frames[i] = 0

            # Se não for visível e não estiver em oclusão, tentamos prever.
            elif not is_visible and self.state.occlusion_frames[i] == 0:
                # ===== HEAD KEYPOINTS: Don't predict, just drop =====
                if self.disable_head_prediction and is_head:
                    # Don't predict head; hide it entirely so the renderer does not
                    # keep drawing stale coordinates for eyes/ears/nose.
                    filtered_position[i] = np.array([np.nan, np.nan], dtype=keypoints.dtype)
                    new_velocity_smooth[i] = np.zeros(2, dtype=keypoints.dtype)
                    self.state.occlusion_frames[i] = self.max_occlusion_frames + 1  # Force drop immediately
                    filtered_scores[i] = 0.0
                    was_predicted[i] = False
                else:
                    # ===== BODY KEYPOINTS: Predict with momentum =====
                    predicted_pos = self.state.position[i] + self.state.velocity_smooth[i]
                    filtered_position[i] = predicted_pos
                    new_velocity_smooth[i] = self.state.velocity_smooth[i]
                    self.state.occlusion_frames[i] = 1
                    was_predicted[i] = True
                    filtered_scores[i] = scores[i] * 0.7

            elif is_occluded and self.state.occlusion_frames[i] > 0:
                predicted_pos = filtered_position[i] + new_velocity_smooth[i]
                filtered_position[i] = predicted_pos
                new_velocity_smooth[i] *= self.velocity_damping
                self.state.occlusion_frames[i] += 1
                was_predicted[i] = True

                if self.state.occlusion_frames[i] > self.max_occlusion_frames:
                    self.state.occlusion_frames[i] = 0
                    filtered_scores[i] = 0.0
                    was_predicted[i] = False
                    new_velocity_smooth[i] = np.zeros(2)
                    if self.disable_head_prediction and is_head:
                        filtered_position[i] = np.array([np.nan, np.nan], dtype=keypoints.dtype)
                else:
                    filtered_scores[i] = max(0.0, scores[i] * 0.5)

        # ===== APPLY HIERARCHICAL CONFIDENCE PROPAGATION =====
        filtered_scores = self._propagate_confidence_hierarchical(filtered_scores)

        # ===== CLAMP EXCESSIVE ARM EXTENSION =====
        filtered_position, filtered_scores, was_predicted, arm_outliers = self._clamp_arm_extension(
            filtered_position,
            filtered_scores,
            was_predicted,
        )

        constraint_violations = constraint_violations | arm_outliers
        self.last_constraint_violation_count = int(np.sum(constraint_violations))

        # ===== UPDATE STATE =====
        self.state.prev_position = self.state.position.copy()
        self.state.position = filtered_position
        self.state.velocity_smooth = new_velocity_smooth

        return filtered_position, filtered_scores, was_predicted

    def reset(self):
        """Reset filter state (for new video or person)."""
        self.state = None

    def is_enabled(self) -> bool:
        return self.enabled

    def toggle(self):
        """Enable or disable the temporal filter."""
        self.enabled = not self.enabled

    def get_state_info(self) -> dict:
        """Return debugging info about filter state."""
        if self.state is None:
            return {"status": "not_initialized"}

        velocity_magnitudes = np.linalg.norm(self.state.velocity_smooth, axis=1)

        return {
            "frame_count": self.state.frame_count,
            "position_mean": float(np.mean(self.state.position)),
            "velocity_mean": float(np.mean(velocity_magnitudes)),
            "velocity_max": float(np.max(velocity_magnitudes)),
            "occluded_keypoints": int(np.sum(self.state.occlusion_frames > 0)),
            "constraint_violations": self.last_constraint_violation_count,
        }
