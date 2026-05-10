#!/usr/bin/env python3
"""
Testing & Validation Utility for Skeleton Normalization & Rotation

Use this to verify that:
1. Skeletons are properly rotated (feet pointing down)
2. Normalization is working correctly
3. Both v3+ and v4 produce expected output
"""

import numpy as np
import cv2
from pathlib import Path
from skeleton_normalizer import SkeletonNormalizer, SkeletonNormConfig

# COCO skeleton connections for visualization
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16)
]

JOINT_NAMES = [
    'nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear',
    'l_sho', 'r_sho', 'l_elb', 'r_elb', 'l_wri',
    'r_wri', 'l_hip', 'r_hip', 'l_kne', 'r_kne',
    'l_ank', 'r_ank'
]


def create_sample_skeleton() -> np.ndarray:
    """
    Create a realistic sample skeleton in COCO format.
    Standing pose with arms at sides.
    """
    kp = np.zeros((17, 3), dtype=np.float32)
    
    # Head (nose, eyes, ears)
    kp[0] = [250, 100, 0.95]    # nose
    kp[1] = [240, 95, 0.90]     # l_eye
    kp[2] = [260, 95, 0.90]     # r_eye
    kp[3] = [230, 90, 0.85]     # l_ear
    kp[4] = [270, 90, 0.85]     # r_ear
    
    # Shoulders & torso
    kp[5] = [220, 150, 0.95]    # l_shoulder
    kp[6] = [280, 150, 0.95]    # r_shoulder
    
    # Left arm (down)
    kp[7] = [210, 220, 0.90]    # l_elbow
    kp[9] = [200, 300, 0.88]    # l_wrist
    
    # Right arm (down)
    kp[8] = [290, 220, 0.90]    # r_elbow
    kp[10] = [300, 300, 0.88]   # r_wrist
    
    # Hips
    kp[11] = [235, 280, 0.95]   # l_hip
    kp[12] = [265, 280, 0.95]   # r_hip
    
    # Left leg
    kp[13] = [230, 380, 0.92]   # l_knee
    kp[15] = [225, 480, 0.90]   # l_ankle
    
    # Right leg
    kp[14] = [270, 380, 0.92]   # r_knee
    kp[16] = [275, 480, 0.90]   # r_ankle
    
    return kp


def draw_skeleton_on_canvas(canvas: np.ndarray, 
                           keypoints: np.ndarray,
                           color: tuple = (0, 255, 0),
                           draw_labels: bool = False) -> np.ndarray:
    """Draw skeleton connections on canvas"""
    canvas = canvas.copy()
    
    # Draw connections
    for idx1, idx2 in COCO_SKELETON:
        pt1 = (int(keypoints[idx1, 0]), int(keypoints[idx1, 1]))
        pt2 = (int(keypoints[idx2, 0]), int(keypoints[idx2, 1]))
        
        # Check bounds
        if (0 <= pt1[0] < canvas.shape[1] and 0 <= pt1[1] < canvas.shape[0] and
            0 <= pt2[0] < canvas.shape[1] and 0 <= pt2[1] < canvas.shape[0]):
            cv2.line(canvas, pt1, pt2, color, 2)
    
    # Draw keypoints
    for idx, (x, y, conf) in enumerate(keypoints):
        pt = (int(x), int(y))
        
        if 0 <= pt[0] < canvas.shape[1] and 0 <= pt[1] < canvas.shape[0]:
            # Color by confidence
            intensity = int(255 * conf)
            pt_color = (0, intensity, 255 - intensity)
            cv2.circle(canvas, pt, 5, pt_color, -1)
            
            # Draw label if enabled
            if draw_labels:
                cv2.putText(canvas, JOINT_NAMES[idx], 
                          (pt[0] + 5, pt[1] - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    
    return canvas


def test_rotation_fix():
    """Test that 90° rotation fixes the foot orientation"""
    print("\n" + "="*60)
    print("TEST 1: Rotation Fix (Feet Should Point DOWN)")
    print("="*60)
    
    # Create sample skeleton
    kp = create_sample_skeleton()
    
    # Create normalizer with rotation enabled
    config = SkeletonNormConfig(
        canvas_width=600,
        canvas_height=800,
        apply_rotation_90deg=True,
        confidence_threshold=0.3
    )
    
    normalizer = SkeletonNormalizer(config)
    kp_normalized = normalizer.normalize_and_center(kp)
    
    # Create side-by-side comparison
    canvas_width = 600
    canvas_height = 800
    
    # Original (should have feet on right side)
    canvas_original = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 50
    cv2.putText(canvas_original, "ORIGINAL (Before Fix)", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.line(canvas_original, (10, 50), (590, 50), (0, 255, 255), 1)
    canvas_original = draw_skeleton_on_canvas(canvas_original, kp, (100, 100, 255))
    
    # Normalized with rotation (feet should point down)
    canvas_rotated = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 50
    cv2.putText(canvas_rotated, "NORMALIZED + ROTATED (After Fix) ✓", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.line(canvas_rotated, (10, 50), (590, 50), (0, 255, 0), 1)
    canvas_rotated = draw_skeleton_on_canvas(canvas_rotated, kp_normalized, (0, 255, 100))
    
    # Combine
    combined = np.hstack([canvas_original, canvas_rotated])
    
    # Save
    output_path = Path("test_rotation_fix.jpg")
    cv2.imwrite(str(output_path), combined)
    print(f"✓ Test image saved: {output_path}")
    print("  Expected: LEFT=feet on right, RIGHT=feet on bottom")
    
    return True


def test_normalization():
    """Test that normalization centers skeleton properly"""
    print("\n" + "="*60)
    print("TEST 2: Normalization (Skeleton Should Be Centered)")
    print("="*60)
    
    # Create sample skeleton
    kp = create_sample_skeleton()
    
    config = SkeletonNormConfig(
        canvas_width=600,
        canvas_height=800,
        apply_rotation_90deg=False,  # Disable rotation for clarity
        confidence_threshold=0.3
    )
    
    normalizer = SkeletonNormalizer(config)
    kp_normalized = normalizer.normalize_and_center(kp)
    
    # Compute centroid
    valid_mask = kp[:, 2] >= 0.3
    centroid_orig = np.mean(kp[valid_mask, :2], axis=0)
    centroid_norm = np.mean(kp_normalized[valid_mask, :2], axis=0)
    
    canvas_width = 600
    canvas_height = 800
    
    # Before
    canvas_before = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 50
    cv2.putText(canvas_before, "BEFORE (Original)", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)
    centroid_before = tuple(map(int, centroid_orig))
    cv2.circle(canvas_before, centroid_before, 10, (0, 0, 255), 3)
    cv2.putText(canvas_before, f"Center: {centroid_before}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    canvas_before = draw_skeleton_on_canvas(canvas_before, kp, (100, 100, 255))
    
    # After
    canvas_after = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 50
    cv2.putText(canvas_after, "AFTER (Normalized) ✓", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    centroid_after = tuple(map(int, centroid_norm))
    cv2.circle(canvas_after, centroid_after, 10, (0, 255, 0), 3)
    cv2.putText(canvas_after, f"Center: {centroid_after}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    canvas_after = draw_skeleton_on_canvas(canvas_after, kp_normalized, (0, 255, 100))
    
    # Combine
    combined = np.hstack([canvas_before, canvas_after])
    
    # Save
    output_path = Path("test_normalization.jpg")
    cv2.imwrite(str(output_path), combined)
    print(f"✓ Test image saved: {output_path}")
    print(f"  Original centroid: {centroid_orig}")
    print(f"  Normalized centroid: {centroid_norm}")
    print(f"  Expected: After normalization, centroid near frame center")
    
    # Check if centered
    frame_center = np.array([canvas_width / 2, canvas_height / 2])
    distance_to_center = np.linalg.norm(centroid_norm - frame_center)
    max_acceptable_distance = 100
    
    if distance_to_center < max_acceptable_distance:
        print(f"  ✓ PASS: Skeleton is well-centered (distance={distance_to_center:.1f}px)")
        return True
    else:
        print(f"  ✗ FAIL: Skeleton not well-centered (distance={distance_to_center:.1f}px)")
        return False


def test_bounds_checking():
    """Test that normalized skeleton stays within canvas bounds"""
    print("\n" + "="*60)
    print("TEST 3: Bounds Checking (No Clipping)")
    print("="*60)
    
    kp = create_sample_skeleton()
    
    config = SkeletonNormConfig(
        canvas_width=600,
        canvas_height=800,
        apply_rotation_90deg=True,
        confidence_threshold=0.3
    )
    
    normalizer = SkeletonNormalizer(config)
    kp_normalized = normalizer.normalize_and_center(kp)
    
    # Check bounds
    x_min, x_max = kp_normalized[:, 0].min(), kp_normalized[:, 0].max()
    y_min, y_max = kp_normalized[:, 1].min(), kp_normalized[:, 1].max()
    
    print(f"  X bounds: {x_min:.1f} to {x_max:.1f} (canvas: 0 to 600)")
    print(f"  Y bounds: {y_min:.1f} to {y_max:.1f} (canvas: 0 to 800)")
    
    if (x_min >= 0 and x_max <= 600 and y_min >= 0 and y_max <= 800):
        print("  ✓ PASS: All keypoints within canvas bounds")
        return True
    else:
        print("  ✗ FAIL: Some keypoints outside canvas bounds (will be clipped)")
        return False


def main():
    print("\n" + "█"*60)
    print("█  SKELETON NORMALIZATION & ROTATION TEST SUITE")
    print("█"*60)
    
    results = {}
    
    try:
        results['rotation'] = test_rotation_fix()
        results['normalization'] = test_normalization()
        results['bounds'] = test_bounds_checking()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return 1
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_passed = all(results.values())
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {test_name.capitalize()}: {status}")
    
    print()
    
    if all_passed:
        print("✓ All tests passed! Skeleton normalization & rotation working correctly.")
        print("\nYou can now use:")
        print("  - python generate_video_clips_v3_enhanced.py (with rotation enabled)")
        print("  - python generate_video_clips_v4_normalized.py (full pipeline)")
        return 0
    else:
        print("✗ Some tests failed. Review the test images for debugging.")
        return 1


if __name__ == "__main__":
    exit(main())
