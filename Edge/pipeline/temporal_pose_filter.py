"""
Temporal Pose Filter
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
    Temporal filter for 17-point COCO skeleton with:
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

        # --- Vectorization pre-computed arrays ---
        self.kinematic_children_arr = np.array(list(self.CHILD_TO_PARENT.keys()))
        self.kinematic_parents_arr = np.array(list(self.CHILD_TO_PARENT.values()))
        
        self.all_kinematic_parents = np.arange(17)
        for c, p in self.CHILD_TO_PARENT.items():
            self.all_kinematic_parents[c] = p
            
        self.lvl1_nodes = np.array([LEFT_ELBOW, RIGHT_ELBOW, LEFT_KNEE, RIGHT_KNEE])
        self.lvl2_nodes = np.array([LEFT_WRIST, RIGHT_WRIST, LEFT_ANKLE, RIGHT_ANKLE])
        
        self.head_mask = np.zeros(17, dtype=bool)
        self.head_mask[list(self.HEAD_KEYPOINTS)] = True

    # ═══════════════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _smoothing_factor(t_e: float, cutoff) -> np.ndarray:
        """1€ smoothing factor α = r / (r + 1), r = 2π·fc·te."""
        r = 2.0 * np.pi * t_e * cutoff
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
                out[children] *= 0.6
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

        parents = self.limb_pairs_arr[:8, 0]
        children = self.limb_pairs_arr[:8, 1]
        
        vecs = pos[children] - pos[parents]
        lengths = np.linalg.norm(vecs, axis=1)
        
        invalid = (lengths > max_len) & (lengths > 1e-6)
        
        if invalid.any():
            inv_p = parents[invalid]
            inv_c = children[invalid]
            inv_vecs = vecs[invalid]
            inv_len = lengths[invalid]
            
            pos[inv_c] = pos[inv_p] + inv_vecs * (max_len / inv_len)[:, None]
            
            overshoot = np.clip(inv_len / max_len - 1.0, 0.0, 1.0)
            scores[inv_c] = np.maximum(0.0, scores[inv_c] * (1.0 - 0.6 * overshoot))
            
            violations[inv_p] = True
            violations[inv_c] = True

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
        valid_vis = st.is_visible
        if valid_vis.any():
            valid_pos = st.position[valid_vis]  # (N, 2)
            valid_kpt = kpts[valid_vis]
            body_speed = float(np.mean(np.linalg.norm(valid_kpt - valid_pos, axis=1)))
        else:
            body_speed = 0.0
        
        body_extra_beta = max(0.0, body_speed - self.body_motion_threshold) * self.body_motion_beta

        # ════════════════════════════════════════════════════════
        # PASS 1 — TRACKING joints (Vectorized)
        # ════════════════════════════════════════════════════════
        tracking_mask = st.is_visible & ~reentry
        
        # Hot-start for reentry
        if reentry.any():
            st.position[reentry] = kpts[reentry]
            st.dx_hat[reentry] = 0.0
            st.state_code[reentry] = TRACKING
            st.occluded_frames[reentry] = 0
            out_pos[reentry] = kpts[reentry]
            out_score[reentry] = raw_s[reentry]
            st.last_conf[reentry] = raw_s[reentry]
            
        if tracking_mask.any():
            conf_trust = raw_s[tracking_mask] ** self.conf_trust_power
            
            raw_dx = (kpts[tracking_mask] - st.position[tracking_mask]) / t_e
            alpha_d = self._smoothing_factor(t_e, self.d_cutoff)
            dx_hat = alpha_d * raw_dx + (1.0 - alpha_d) * st.dx_hat[tracking_mask]
            st.dx_hat[tracking_mask] = dx_hat
            
            speed = np.linalg.norm(dx_hat, axis=1)
            effective_beta = (self.beta * self._joint_beta_mult[tracking_mask]) + body_extra_beta
            cutoff = self.min_cutoff + effective_beta * speed
            
            alpha_base = self._smoothing_factor(t_e, cutoff)
            alpha = alpha_base * conf_trust + 0.02
            alpha = alpha[:, None]  # broadcast to (N, 2)
            
            out_pos[tracking_mask] = alpha * kpts[tracking_mask] + (1.0 - alpha) * st.position[tracking_mask]
            out_score[tracking_mask] = raw_s[tracking_mask]
            
            st.state_code[tracking_mask] = TRACKING
            st.occluded_frames[tracking_mask] = 0
            st.last_conf[tracking_mask] = raw_s[tracking_mask]

        # ── Update kinematic offsets for all TRACKING pairs ───────────
        c_arr = self.kinematic_children_arr
        p_arr = self.kinematic_parents_arr
        valid_kin = st.is_visible[c_arr] & st.is_visible[p_arr]
        
        if valid_kin.any():
            c_idx = c_arr[valid_kin]
            p_idx = p_arr[valid_kin]
            
            new_offset = out_pos[c_idx] - out_pos[p_idx]
            new_rel_dx = new_offset - st.rel_offset[c_idx]
            alpha_rd = self._smoothing_factor(t_e, self.d_cutoff * 0.5)
            
            st.rel_dx[c_idx] = alpha_rd * new_rel_dx + (1.0 - alpha_rd) * st.rel_dx[c_idx]
            st.rel_offset[c_idx] = new_offset

        # ════════════════════════════════════════════════════════
        # PASS 2 — OCCLUDED / LOST joints (Vectorized)
        # ════════════════════════════════════════════════════════
        not_visible = ~st.is_visible
        
        invalid_head = not_visible & self.head_mask & self.disable_head_prediction
        if invalid_head.any():
            st.state_code[invalid_head] = LOST
            st.occluded_frames[invalid_head] = self.max_occlusion_frames + 1
            st.position[invalid_head] = np.nan
            st.dx_hat[invalid_head] = 0.0
            out_pos[invalid_head] = np.nan
            out_score[invalid_head] = 0.0
            
        nan_pos = not_visible & np.isnan(st.position).any(axis=1) & ~invalid_head
        if nan_pos.any():
            st.state_code[nan_pos] = LOST
            st.occluded_frames[nan_pos] = self.max_occlusion_frames + 1
            out_pos[nan_pos] = np.nan
            out_score[nan_pos] = 0.0

        proc_mask = not_visible & ~invalid_head & ~nan_pos
        st.occluded_frames[proc_mask] += 1
        
        lost_mask = proc_mask & (st.occluded_frames > self.max_occlusion_frames)
        if lost_mask.any():
            st.state_code[lost_mask] = LOST
            out_pos[lost_mask] = st.position[lost_mask]
            out_score[lost_mask] = 0.0
            predicted[lost_mask] = False
            
        occ_mask = proc_mask & (st.occluded_frames <= self.max_occlusion_frames)
        if occ_mask.any():
            st.state_code[occ_mask] = OCCLUDED
            predicted[occ_mask] = True
            
            root_mask = occ_mask.copy()
            root_mask[self.kinematic_children_arr] = False
            
            if root_mask.any():
                out_pos[root_mask] = st.position[root_mask] + st.dx_hat[root_mask]
                st.dx_hat[root_mask] *= self.velocity_damping
                
            for lvl_nodes in [self.lvl1_nodes, self.lvl2_nodes]:
                lvl_mask = occ_mask.copy()
                lvl_mask_idx = [i for i in range(17) if i not in lvl_nodes]
                lvl_mask[lvl_mask_idx] = False
                
                if lvl_mask.any():
                    st.rel_offset[lvl_mask] += st.rel_dx[lvl_mask]
                    st.rel_dx[lvl_mask] *= self.rel_vel_damping
                    
                    parents_of_lvl = self.all_kinematic_parents[lvl_mask]
                    out_pos[lvl_mask] = out_pos[parents_of_lvl] + st.rel_offset[lvl_mask]
                    st.dx_hat[lvl_mask] *= self.velocity_damping

            t_occ = st.occluded_frames[occ_mask]
            out_score[occ_mask] = np.maximum(0.0, st.last_conf[occ_mask] * (self.occlusion_conf_decay ** t_occ))

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
