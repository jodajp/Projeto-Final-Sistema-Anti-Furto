"""
Temporal Pose Filter — Research-Grade Edition
==============================================

Architecture:
  1. **Confidence-Weighted 1€ Filter** (per keypoint):
     - Low-pass cutoff adapts to instantaneous velocity (beta term).
     - Measurement trust is modulated by RTMO confidence — less trust → more smoothing.
     - Prevents high-confidence hallucinations from dominating the estimate.

  2. **Kinematic Occlusion Prediction** (skeletal chain):
     - On occlusion, child joints (wrist, elbow, knee, ankle) are predicted relative
       to their parent joint using the last known relative offset + smoothed relative
       velocity. This eliminates the "mid-air freeze" / "rubber band drag" effect.
     - Relative velocity is itself 1€-filtered to avoid noisy predictions.

  3. **Clean 3-State Machine per keypoint**:
     - TRACKING  : keypoint observed with adequate confidence — apply 1€ filter.
     - OCCLUDED  : keypoint hidden — use kinematic chain prediction.
     - LOST      : occlusion exceeded window — drop keypoint / head to NaN.

  4. **Re-entry Hot-Start Guard**:
     - When a keypoint reappears after occlusion, the filter state is hot-started
       at the predicted position so the 1€ filter doesn't blend from a stale past.

  5. **Exponential Confidence Decay During Occlusion**:
     - Occluded joints smoothly fade: conf(t) = last_conf * decay^t
     - Produces natural visual fade rather than sudden pop-in/pop-out.

References:
  - Casiez et al. (2012), "1€ Filter: A Simple Speed-Based Low-Pass Filter for
    Noisy Input in Interactive Systems."
  - CBKF: Confidence-Based Kalman Filter for Pose Tracking (pattern recognition lit.)
  - PoseFlow / LightTrack: Kinematic link-based temporal smoothing.
"""

from typing import Any, Optional, Tuple
import numpy as np
import math
from Detecao.skeleton import (
    NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR,
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
    DEFAULT_LIMBS
)

# ─── State constants ──────────────────────────────────────────────────
TRACKING = 0
OCCLUDED = 1
LOST     = 2


class AdaptiveFilterState:
    """Per-track filter state for all 17 keypoints."""

    NUM_KP = 17

    def __init__(self, keypoints: np.ndarray, scores: np.ndarray, confidence_high: float):
        n = self.NUM_KP
        # 1€ filter primary state
        self.position = keypoints.copy().astype(np.float64)    # (17, 2) filtered position
        self.dx_hat   = np.zeros((n, 2), dtype=np.float64)     # (17, 2) filtered velocity

        # Hysteresis visibility
        self.is_visible = (scores >= confidence_high)           # (17,) bool

        # Occlusion state machine
        self.state_code      = np.where(self.is_visible, TRACKING, LOST).astype(np.int8)
        self.occluded_frames = np.zeros(n, dtype=np.int32)

        # Kinematic chain: relative child-parent offset and its smoothed velocity
        self.rel_offset = np.zeros((n, 2), dtype=np.float64)   # child − parent (image space)
        self.rel_dx     = np.zeros((n, 2), dtype=np.float64)   # smoothed delta of rel_offset

        # Last observed raw confidence (for exponential decay during occlusion)
        self.last_conf = scores.astype(np.float64).clip(0, 1)  # (17,)

        self.frame_count = 0


class TemporalPoseFilter:
    """
    Research-grade temporal filter for 17-point COCO skeleton with:
    - Confidence-weighted 1€ adaptive smoothing
    - Kinematic chain occlusion prediction
    - Clean 3-state machine (TRACKING / OCCLUDED / LOST)
    - Re-entry hot-start guard
    - Exponential confidence decay for smooth visual fade
    """

    LIMB_PAIRS = DEFAULT_LIMBS

    SKELETON_TREE = {
        LEFT_SHOULDER:  [LEFT_ELBOW, LEFT_HIP],
        RIGHT_SHOULDER: [RIGHT_ELBOW, RIGHT_HIP],
        LEFT_ELBOW:     [LEFT_WRIST],
        RIGHT_ELBOW:    [RIGHT_WRIST],
        LEFT_HIP:       [LEFT_KNEE],
        RIGHT_HIP:      [RIGHT_KNEE],
        LEFT_KNEE:      [LEFT_ANKLE],
        RIGHT_KNEE:     [RIGHT_ANKLE],
    }

    # Child → parent mapping for kinematic prediction
    CHILD_TO_PARENT = {
        LEFT_ELBOW:   LEFT_SHOULDER,
        RIGHT_ELBOW:  RIGHT_SHOULDER,
        LEFT_WRIST:   LEFT_ELBOW,
        RIGHT_WRIST:  RIGHT_ELBOW,
        LEFT_KNEE:    LEFT_HIP,
        RIGHT_KNEE:   RIGHT_HIP,
        LEFT_ANKLE:   LEFT_KNEE,
        RIGHT_ANKLE:  RIGHT_KNEE,
    }

    # Topological order: parents must be processed before children
    TOPOLOGICAL_ORDER = [
        LEFT_ELBOW,  RIGHT_ELBOW,  LEFT_KNEE,   RIGHT_KNEE,
        LEFT_WRIST,  RIGHT_WRIST,  LEFT_ANKLE,  RIGHT_ANKLE,
    ]

    HEAD_KEYPOINTS = frozenset([NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR])

    def __init__(self, config_source, frame_skip: Optional[int] = None):
        raw: dict[str, Any] = {}
        if config_source is not None:
            getter = getattr(config_source, "temporal_filter_config", None)
            if callable(getter):
                result = getter() or {}
                if isinstance(result, dict):
                    raw = result
            elif isinstance(config_source, dict):
                raw = config_source

        self.enabled = bool(raw.get("enabled", True))
        
        # Determine frame_skip (prioritize parameter, fallback to config runtime)
        self.frame_skip = 1
        if frame_skip is not None:
            self.frame_skip = int(frame_skip)
        elif config_source is not None and hasattr(config_source, "data"):
            self.frame_skip = int(config_source.data.get("runtime", {}).get("frame_skip", 1))

        # 1€ Filter — tune these to control lag vs jitter tradeoff
        self.min_cutoff = float(raw.get("one_euro_min_cutoff", 0.8))
        self.beta       = float(raw.get("one_euro_beta",       0.08))
        self.d_cutoff   = float(raw.get("one_euro_d_cutoff",   1.0))

        # Confidence Hysteresis
        self.confidence_high = float(raw.get("confidence_high", 0.40))
        self.confidence_low  = float(raw.get("confidence_low",  0.15))

        # Occlusion (Dampings and decay are raised to the power of frame_skip to preserve physical time dynamics)
        self.max_occlusion_frames = int(  raw.get("max_occlusion_frames", 15))
        self.velocity_damping     = float(raw.get("velocity_damping",     0.92)) ** self.frame_skip
        self.rel_vel_damping      = float(raw.get("rel_vel_damping",      0.88)) ** self.frame_skip
        self.occlusion_conf_decay = float(raw.get("occlusion_conf_decay", 0.75)) ** self.frame_skip

        # Confidence trust: alpha = alpha_base * score^power  (lower = more smoothing for weak detections)
        self.conf_trust_power = float(raw.get("conf_trust_power", 0.5))

        # Body rotation compensation:
        # When the whole body moves fast (e.g. person turns), open up the filter for ALL joints
        # so arms don't lag behind. body_motion_beta is added on top of per-joint beta when
        # the body CoM speed exceeds body_motion_threshold pixels/frame.
        self.body_motion_beta      = float(raw.get("body_motion_beta",      0.30))
        self.body_motion_threshold = float(raw.get("body_motion_threshold", 8.0))

        # Per-joint beta multipliers: distal joints (wrists/ankles) react faster
        # than proximal joints (shoulders/hips) to follow the body during fast turns.
        self.distal_beta_mult = float(raw.get("distal_beta_mult", 2.5))
        _D = {
            LEFT_WRIST: self.distal_beta_mult, RIGHT_WRIST: self.distal_beta_mult,
            LEFT_ANKLE: self.distal_beta_mult, RIGHT_ANKLE: self.distal_beta_mult,
            LEFT_ELBOW: 1.5,                   RIGHT_ELBOW: 1.5,
            LEFT_KNEE:  1.5,                   RIGHT_KNEE:  1.5,
        }
        self._joint_beta_mult = np.array([_D.get(i, 1.0) for i in range(17)], dtype=np.float64)

        # Hierarchical confidence propagation
        self.enable_hierarchical_confidence = bool(raw.get("enable_hierarchical_confidence", True))
        self.occlusion_confidence_threshold = float(raw.get("occlusion_confidence_threshold", 0.35))

        # Head handling
        self.disable_head_prediction = bool(raw.get("disable_head_prediction", True))

        # Skeletal constraint: max limb length as multiple of torso scale
        self.limb_scale_max = float(raw.get("limb_stretch_threshold", 1.2))

        self.limb_pairs_arr = np.asarray(self.LIMB_PAIRS)
        self.last_constraint_violations = 0
        self.state: Optional[AdaptiveFilterState] = None

    # ═══════════════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _smoothing_factor(t_e: float, cutoff) -> np.ndarray:
        """1€ smoothing factor α = r / (r + 1), r = 2π·fc·te."""
        r = 2.0 * math.pi * t_e * cutoff
        return r / (r + 1.0)

    def _get_person_scale(self, kpts: np.ndarray) -> float:
        """Robust body scale estimate from torso + shoulder/hip widths."""
        neck   = (kpts[LEFT_SHOULDER] + kpts[RIGHT_SHOULDER]) * 0.5
        pelvis = (kpts[LEFT_HIP]      + kpts[RIGHT_HIP]     ) * 0.5
        torso  = float(np.linalg.norm(neck - pelvis))
        shldr  = float(np.linalg.norm(kpts[LEFT_SHOULDER] - kpts[RIGHT_SHOULDER]))
        hip    = float(np.linalg.norm(kpts[LEFT_HIP]      - kpts[RIGHT_HIP]     ))
        return max(torso, shldr * 1.5, hip * 1.5, 30.0)

    def _propagate_confidence(self, scores: np.ndarray) -> np.ndarray:
        """Dampen children's confidence when their parent joint is uncertain."""
        if not self.enable_hierarchical_confidence:
            return scores.copy()
        out = scores.copy()
        for parent, children in self.SKELETON_TREE.items():
            if out[parent] < self.occlusion_confidence_threshold:
                for child in children:
                    out[child] *= 0.6
        return out

    def _enforce_limb_lengths(
        self,
        pos: np.ndarray,
        scores: np.ndarray,
        predicted: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Project limb joints back onto anatomically plausible positions.
        Uses smooth score damping proportional to overshoot (not a hard jump).
        """
        scale      = self._get_person_scale(pos)
        max_len    = scale * self.limb_scale_max
        violations = np.zeros(17, dtype=bool)

        for parent, child in self.LIMB_PAIRS[:8]:  # arms + legs only
            vec    = pos[child] - pos[parent]
            length = float(np.linalg.norm(vec))
            if length <= 1e-6 or length <= max_len:
                continue
            pos[child]    = pos[parent] + vec * (max_len / length)
            overshoot     = min(length / max_len - 1.0, 1.0)
            scores[child] = max(0.0, scores[child] * (1.0 - 0.6 * overshoot))
            violations[parent] = violations[child] = True

        return pos, scores, predicted, violations

    # ═══════════════════════════════════════════════════════════════════
    # MAIN FILTER
    # ═══════════════════════════════════════════════════════════════════

    def filter_pose(
        self, keypoints: np.ndarray, scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply temporal filtering to a single person's pose.

        Args:
            keypoints : (17, 2) raw RTMO keypoints in image space.
            scores    : (17,)   raw RTMO confidence scores [0, 1].

        Returns:
            filtered_pos    : (17, 2) smoothed keypoints (float32).
            filtered_scores : (17,)   output confidence values (float32).
            was_predicted   : (17,)   True where position was extrapolated.
        """
        if not self.enabled:
            return keypoints.copy(), scores.copy(), np.zeros(17, dtype=bool)

        # ── Initialise on first frame ─────────────────────────────────
        if self.state is None:
            self.state = AdaptiveFilterState(keypoints, scores, self.confidence_high)
            return keypoints.copy(), scores.copy(), np.zeros(17, dtype=bool)

        st = self.state
        st.frame_count += 1
        t_e = float(self.frame_skip)  # dt matches frame_skip to preserve physical time steps

        kpts  = keypoints.astype(np.float64)
        raw_s = scores.astype(np.float64).clip(0.0, 1.0)

        # ── Output buffers ────────────────────────────────────────────
        out_pos   = st.position.copy()
        out_score = np.zeros(17, dtype=np.float64)
        predicted = np.zeros(17, dtype=bool)

        # ── Hysteresis visibility ─────────────────────────────────────
        was_visible = st.is_visible.copy()
        st.is_visible = np.where(
            raw_s >= self.confidence_high, True,
            np.where(raw_s < self.confidence_low, False, was_visible)
        )

        # Re-entry: was LOST/OCCLUDED last frame, now TRACKING
        reentry = st.is_visible & ~was_visible

        # Head mask
        is_head = np.zeros(17, dtype=bool)
        for h in self.HEAD_KEYPOINTS:
            is_head[h] = True

        # ── Body motion speed (CoM velocity) ─────────────────────────
        # Compute how fast the body's centre of mass is moving this frame.
        # When the person turns, ALL visible joints translate quickly.
        # We use this to add extra beta (open the filter wider) for all joints
        # so arms don't lag behind during body rotations.
        valid_vis = st.is_visible
        if valid_vis.any():
            valid_pos = st.position[valid_vis]  # (N, 2)
            valid_kpt = kpts[valid_vis]
            body_speed = float(np.mean(np.linalg.norm(valid_kpt - valid_pos, axis=1)))
        else:
            body_speed = 0.0
        # Extra beta is proportional to how far above threshold we are
        body_extra_beta = max(0.0, body_speed - self.body_motion_threshold) * self.body_motion_beta

        # ════════════════════════════════════════════════════════
        # PASS 1 — TRACKING joints
        # ════════════════════════════════════════════════════════
        for idx in range(17):
            if not st.is_visible[idx]:
                continue

            if reentry[idx]:
                # Hot-start: seed at current detector position to avoid blending
                # from a potentially stale position stored before occlusion.
                st.position[idx]      = kpts[idx]
                st.dx_hat[idx]        = np.zeros(2)
                st.state_code[idx]    = TRACKING
                st.occluded_frames[idx] = 0
                out_pos[idx]          = kpts[idx]
                out_score[idx]        = raw_s[idx]
                st.last_conf[idx]     = raw_s[idx]
                continue

            # ── Confidence-weighted 1€ filter with body-motion boost ───
            conf_trust = float(raw_s[idx]) ** self.conf_trust_power

            raw_dx  = (kpts[idx] - st.position[idx]) / t_e
            alpha_d = self._smoothing_factor(t_e, self.d_cutoff)
            dx_hat  = alpha_d * raw_dx + (1.0 - alpha_d) * st.dx_hat[idx]
            st.dx_hat[idx] = dx_hat

            speed  = float(np.linalg.norm(dx_hat))
            # Per-joint beta * global body-motion boost = fast during turns, smooth when still
            effective_beta = (self.beta * self._joint_beta_mult[idx]) + body_extra_beta
            cutoff = self.min_cutoff + effective_beta * speed

            alpha_base = self._smoothing_factor(t_e, cutoff)
            alpha = float(alpha_base) * conf_trust + 0.02  # 2% floor

            out_pos[idx]   = alpha * kpts[idx] + (1.0 - alpha) * st.position[idx]
            out_score[idx] = raw_s[idx]

            st.state_code[idx]      = TRACKING
            st.occluded_frames[idx] = 0
            st.last_conf[idx]       = raw_s[idx]

        # ── Update kinematic offsets for all TRACKING pairs ───────────
        for child_idx, parent_idx in self.CHILD_TO_PARENT.items():
            if st.is_visible[child_idx] and st.is_visible[parent_idx]:
                new_offset = out_pos[child_idx] - out_pos[parent_idx]
                new_rel_dx = new_offset - st.rel_offset[child_idx]
                # Smooth relative velocity to avoid single-frame noise spikes
                alpha_rd = self._smoothing_factor(t_e, self.d_cutoff * 0.5)
                st.rel_dx[child_idx]     = (
                    alpha_rd * new_rel_dx + (1.0 - alpha_rd) * st.rel_dx[child_idx]
                )
                st.rel_offset[child_idx] = new_offset

        # ════════════════════════════════════════════════════════
        # PASS 2 — OCCLUDED / LOST joints (topological order)
        # ════════════════════════════════════════════════════════
        for idx in range(17):
            if st.is_visible[idx]:
                continue

            # Head keypoints: never predict — drop to NaN
            if is_head[idx] and self.disable_head_prediction:
                st.state_code[idx]       = LOST
                st.occluded_frames[idx]  = self.max_occlusion_frames + 1
                st.position[idx]         = np.full(2, np.nan)
                st.dx_hat[idx]           = np.zeros(2)
                out_pos[idx]             = np.full(2, np.nan)
                out_score[idx]           = 0.0
                continue

            # If we have no valid position to predict from, mark LOST
            if np.isnan(st.position[idx]).any():
                st.state_code[idx]       = LOST
                st.occluded_frames[idx]  = self.max_occlusion_frames + 1
                out_pos[idx]             = np.full(2, np.nan)
                out_score[idx]           = 0.0
                continue

            st.occluded_frames[idx] += 1

            if st.occluded_frames[idx] > self.max_occlusion_frames:
                # Window expired → mark LOST, expose last known position quietly
                st.state_code[idx] = LOST
                out_pos[idx]       = st.position[idx]
                out_score[idx]     = 0.0
                predicted[idx]     = False
            else:
                st.state_code[idx] = OCCLUDED
                predicted[idx]     = True

                if idx in self.CHILD_TO_PARENT:
                    # ── Kinematic prediction relative to parent ──────
                    parent_idx = self.CHILD_TO_PARENT[idx]
                    # Advance relative offset by its own smoothed velocity
                    st.rel_offset[idx] = st.rel_offset[idx] + st.rel_dx[idx]
                    st.rel_dx[idx]     = st.rel_dx[idx] * self.rel_vel_damping
                    predicted_pos      = out_pos[parent_idx] + st.rel_offset[idx]
                else:
                    # Root joint (shoulder/hip): absolute velocity prediction
                    predicted_pos = st.position[idx] + st.dx_hat[idx]

                st.dx_hat[idx] = st.dx_hat[idx] * self.velocity_damping
                out_pos[idx]   = predicted_pos

                # Exponential decay: conf(t) = last_conf * decay^frames_occluded
                t_occ = st.occluded_frames[idx]
                out_score[idx] = max(0.0, st.last_conf[idx] * (self.occlusion_conf_decay ** t_occ))

        # ════════════════════════════════════════════════════════
        # POST-PROCESSING
        # ════════════════════════════════════════════════════════

        # Sync valid positions back to state
        valid = ~np.isnan(out_pos).any(axis=1)
        st.position[valid] = out_pos[valid]

        # Hierarchical confidence propagation
        out_score = self._propagate_confidence(out_score)

        # Anatomical limb-length enforcement
        out_pos, out_score, predicted, violations = self._enforce_limb_lengths(
            out_pos, out_score, predicted
        )
        self.last_constraint_violations = int(np.sum(violations))

        # Sync again after clamping
        valid2 = ~np.isnan(out_pos).any(axis=1)
        st.position[valid2] = out_pos[valid2]

        return out_pos.astype(np.float32), out_score.astype(np.float32), predicted

    # ═══════════════════════════════════════════════════════════════════
    # API
    # ═══════════════════════════════════════════════════════════════════

    def reset(self):
        self.state = None

    def is_enabled(self) -> bool:
        return self.enabled

    def toggle(self):
        self.enabled = not self.enabled

    def get_state_info(self) -> dict:
        if self.state is None:
            return {"status": "not_initialized"}
        st = self.state
        return {
            "frame_count":           st.frame_count,
            "tracked_keypoints":     int(np.sum(st.state_code == TRACKING)),
            "occluded_keypoints":    int(np.sum(st.state_code == OCCLUDED)),
            "lost_keypoints":        int(np.sum(st.state_code == LOST)),
            "velocity_mean":         float(np.nanmean(np.linalg.norm(st.dx_hat, axis=1))),
            "constraint_violations": self.last_constraint_violations,
        }
