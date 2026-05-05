# NEXT STEPS ACTION PLAN
## RetailS Dataset - Data Cleaning & Continuous Improvement

**Current Status:** Dataset analysis complete, improvement opportunities identified  
**Team Focus:** Data cleaning + self-learning implementation  
**Timeline:** Short-term wins first (this week), then medium-term improvements (next 2 weeks)

---

## IMMEDIATE ACTIONS (This Week) ✅

### ACTION 1: Verify All Data is Correctly Indexed
**File:** `Visualizar_Data/build_complete_manifest.py`  
**Command:**
```bash
cd Visualizar_Data
python build_complete_manifest.py
```
**Expected output:** 
- 1,613 JSON files found ✓
- 36,555 normal sequences confirmed ✓
- 896 suspicious sequences confirmed ✓

**Why:** Ensure no files are missing before improvement work

---

### ACTION 2: Clean Repository - Remove Unnecessary Files
**Files to delete (safe to remove):**
```
Visualizar_Data/
├── Suspicious_Dataset/                  # ← Old format, redundant with Manifests/
├── build_suspicious_dataset.py           # ← Legacy
├── build_suspicious_manifest.py          # ← Superseded by v2
├── gt_viewer.py                          # ← Debug tool only
├── test_setup.py                         # ← Testing only
├── visualize.py                          # ← Functions now in v2
└── analyze_json.py, inspect_data.py      # ← Analysis temp files
```

**Equivalent save:** ~15-20 MB  
**Command:**
```bash
cd Visualizar_Data
rm -r Suspicious_Dataset/
rm build_suspicious_dataset.py build_suspicious_manifest.py gt_viewer.py
```

**Caution:** Keep these (still needed):
```
✅ generate_video_clips.py           # Legacy - works, documented
✅ generate_video_clips_v2.py        # Enhanced - confidence filtering + interpolation
✅ generate_video_clips_v3_enhanced.py  # New - hand detection + motion analysis
✅ build_complete_manifest.py        # Data indexer
✅ Manifests/*.json                  # (6 files) - Critical data
✅ Data/                             # (14 GB) - Raw dataset
```

---

### ACTION 3: Generate Data Quality Report
**Create file:** `Visualizar_Data/data_quality_report.py`
```python
import json
import numpy as np
from pathlib import Path

# For EACH video file:
# - Count frames with confidence > 0.7
# - Count detected people per frame
# - Verify GT labels exist
# - Flag any anomalies

# Output: CSV with quality metrics
```

**Why:** Identify problematic files before training

---

## SHORT-TERM IMPROVEMENTS (Week 1-2) 📊

### IMPROVEMENT 1: Add Confidence Filtering to v2
**File to modify:** `generate_video_clips_v2.py`  
**Change required:** 2 lines

```python
# LINE 150-160: In normalize_skeleton_to_frame()

# BEFORE:
x_coords = valid_kps[:, 0]
y_coords = valid_kps[:, 1]

# AFTER: (add confidence threshold)
confidence_threshold = 0.5  # Only use joints with conf > 50%
valid_joints = kp_array[:, 2] >= confidence_threshold
valid_kps = kp_array[valid_joints]
x_coords = valid_kps[:, 0]
y_coords = valid_kps[:, 1]
```

**Impact:** Cleaner skeletons, fewer visualization errors  
**Testing:**
```bash
python generate_video_clips_v2.py --normal 3 --suspicious 3
# Check output videos - skeletons should be cleaner
```

---

### IMPROVEMENT 2: Test Enhanced Visualizer v3
**File:** `generate_video_clips_v3_enhanced.py` (already provided)

**Features included:**
- ✅ Confidence-based joint coloring (green=high, red=low)
- ✅ Hand-pocket proximity detection
- ✅ Motion vector visualization
- ✅ Ground truth label overlay

**Test it:**
```bash
python generate_video_clips_v3_enhanced.py --normal 2 --suspicious 2
cd output_v3
# Review videos - do confidence colors help?
#  - Can you see which joints are unreliable?
#  - Do hand risk indicators make sense?
```

---

### IMPROVEMENT 3: Create Comparison Report
**Compare outputs from v2 vs v3:**
```
Aspect                  | v2        | v3
Skeleton Quality        | Good      | Excellent (confidence-based)
Hand Detection          | No        | Yes ✓
Motion Vectors          | No        | Yes ✓
Suspicious Highlighting | Labels    | Labels + Risk Scores
Useful for Training     | Yes       | Yes + Features
```

**Document findings** in `IMPROVEMENT_ANALYSIS.md`

---

## MEDIUM-TERM IMPROVEMENTS (Week 3-4) 🎯

### IMPROVEMENT 4: Hand-Pocket Anomaly Detector
**Create file:** `Atividades_Suspeitas/hand_proximity_detector.py`

```python
class HandProximityDetector:
    """
    Detects when hands approach pocket/chest area
    Risk score: 0.0 (safe) to 1.0 (hand on pocket)
    """
    
    def __init__(self, pocket_region=(200, 1000, 600, 1200)):
        self.pocket_region = pocket_region
    
    def detect(self, skeleton_frame):
        """
        Input: 17-joint skeleton for one frame
        Output: risk_score (0-1)
        """
        # Get hand joints (COCO indices 9, 10, 11 = wrists)
        # Get torso joint (center of shoulders/hips)
        # Calculate distance
        # Return: 1.0 - (distance / max_distance)
        pass

# Similar for face (concealment, removing mask, etc.)
```

**Integration:** Use in pipeline early-stage filtering  
**Expected impact:** 30% false positive reduction

---

### IMPROVEMENT 5: Motion Analysis Module
**Create file:** `Atividades_Suspeitas/motion_analyzer.py`

```python
class MotionAnalyzer:
    """
    Detects erratic/unusual movements
    Input: skeleton sequence (multiple frames)
    Output: Motion anomaly score per frame
    """
    
    def analyze(self, skeleton_sequence):
        """
        1. Calculate velocity: (kp_t - kp_t-1)
        2. Calculate acceleration: (vel_t - vel_t-1)
        3. Flag sudden changes
        4. Return: abnormality_score (0-1)
        """
        pass
```

**Why:** Shoplifting involves jerky/purposeful movements  
**Expected impact:** Detect 40-60% of suspicious activities

---

## SELF-LEARNING & CONTINUOUS IMPROVEMENT 🔄

### Phase 1: Baseline Metrics
Establish current performance:
```python
# For validation set (100 videos):
- Confidence filtering impact: % reduction in artifacts
- Hand detection: % frames correctly identified as risky
- Motion detection: % true positive anomalies
- Overall false positive rate

# Benchmark: "Before implementing improvements"
```

### Phase 2: Iterative Refinement
Roll out improvements in order:
1. Confidence filtering → Measure impact
2. Hand detection → Add and measure
3. Motion analysis → Add and measure
4. Combination scoring → Final model

### Phase 3: Validation
On held-out test set:
```
Metric                      Target      Current
Sensitivity (catch theft)    >75%        ?
Specificity (normal people)  >90%        ?
F1-Score                     >0.80       ?
```

---

## INFRASTRUCTURE FOR CONTINUOUS IMPROVEMENT

### Create Model Registry
**File:** `pipeline/models_log.json`
```json
{
  "models": [
    {
      "name": "confidence_filter_v1",
      "date": "2026-04-14",
      "features": ["pose_confidence"],
      "metrics": {"accuracy": 0.85, "f1": 0.78},
      "notes": "Filters low-confidence joints"
    },
    {
      "name": "hand_detector_v1",
      "date": "2026-04-21",
      "features": ["hand_proximity", "pocket_distance"],
      "metrics": {"sensitivity": 0.65, "specificity": 0.92}
    }
  ]
}
```

### Create Evaluation Pipeline
**File:** `pipeline/evaluate.py`
```python
def evaluate_detector(test_videos, detector_model):
    """
    Run detection on test set
    Return: TP, FP, TN, FN, accuracy, precision, recall, F1
    """
    pass

# Run this after each improvement
# Track metrics over time
```

---

## DECISION MATRIX: What to Do When

| Situation | Action |
|-----------|--------|
| **Want better visualization?** | Use v3_enhanced (has all features) |
| **Training a model?** | Start with v2 + confidence filtering |
| **Detecting anomalies in real-time?** | Build on v3_enhanced + hand detector |
| **Performing research analysis?** | Use data_quality_report.py |
| **Storage/cleanup?** | Remove Suspicious_Dataset/, legacy scripts |
| **Comparison studies?** | Run v2 vs v3 on same videos |

---

## QUESTIONS & ANSWERS

**Q: Do we have item location info?**  
A: No, not in dataset. But we can infer via hand proximity + motion anomalies.

**Q: Should we keep the old scripts?**  
A: Keep generate_video_clips.py (documented), delete others (archived if needed).

**Q: How long until "self-learning" system?**  
A: Week 4 - build iterative evaluation pipeline that tracks model improvements.

**Q: Priority - v2 improvements or v3?**  
A: Test v3_enhanced first (has everything), then backport good ideas to v2.

**Q: Can confidence scores fix the skeleton rotation issue?**  
A: Partially - filtering low-conf joints reduces visual errors. Already in v3.

---

## SUCCESS CRITERIA

By end of **Week 4**, you should have:

- ✅ Data quality report (week 1)
- ✅ Repository cleaned (week 1)
- ✅ v3_enhanced tested and validated (week 2)
- ✅ Hand-proximity detector implemented (week 3)
- ✅ Motion analysis module working (week 3)
- ✅ Evaluation framework in place (week 4)
- ✅ First iteration of "self-learning" metrics recorded (week 4)

---

## FILES TO TRACK

```
Visualizar_Data/
├── DATA_AUDIT_COMPLETE.md              ← Read this for detailed analysis
├── generate_video_clips_v2.py           ← Add confidence filtering
├── generate_video_clips_v3_enhanced.py  ← Test & evaluate
├── data_quality_report.py               ← Create & run (ACTION 3)
└── Manifests/manifest_all.json          ← Verified complete

Atividades_Suspeitas/
├── hand_proximity_detector.py           ← Create (IMPROVEMENT 4)
├── motion_analyzer.py                   ← Create (IMPROVEMENT 5)
└── (other activity detectors)

pipeline/
├── models_log.json                      ← Track improvements over time
└── evaluate.py                          ← Continuous evaluation
```

---

## SUMMARY

**What's Working:** Dataset parsing, video generation, visualization  
**What's Needed:** Data cleaning, feature engineering, continuous evaluation  
**What's Missing:** Advanced ML (but framework ready for it)  

The system is ready for **self-learning** - you now have:
1. ✅ Complete dataset (1,613 files, 36K+ sequences)
2. ✅ Baseline visualization (v2 + v3)
3. ✅ Feature opportunities identified (confidence, hand-pocket, motion)
4. ✅ Framework for iterative improvement (proposed)

**Start with ACTION 1 today** (verify manifests), then move to IMPROVEMENTS in order.

---

*Generated: 2026-04-14 | Next Review: 2026-04-21*

