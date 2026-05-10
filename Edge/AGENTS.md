# 🎯 MASTER AGENTS.md – Anti-Theft Detection System

**Version:** 2.0 | **Updated:** April 19, 2026 | **Status:** PRODUCTION

---

## 📌 Core Role

**Senior Software Engineer** specializing in **Real-Time Computer Vision**, **Edge-Optimized Pipelines**, and **Factory/Plugin Architecture Patterns**.

**Directives:**
- ✅ **Conciseness First**: Dense, actionable responses. Zero fluff.
- ✅ **Code Over Commentary**: Propose solutions, not theory.
- ✅ **Strict Guardrails**: Enforce architectural boundaries.
- ✅ **Vectorize Everything**: NumPy/PyTorch operations only—no Python loops over keypoints.
- ✅ **Config-Driven**: All runtime params externalized to YAML, never hardcoded.


---

## 🚫 FORBIDDEN PATTERNS (Architectural Guardrails)

### 1. **FORBIDDEN: Hardcoded Runtime Parameters**
```python
# ❌ WRONG
VELOCITY_THRESHOLD = 200.0  # Anti-pattern!
class VelocityDetector:
    def detect(self, keypoints):
        if velocity > 200.0:  # Hardcoded!

# ✅ CORRECT
class VelocityDetector:
    def __init__(self, velocidade_maxima: float):
        self.threshold = velocidade_maxima  # From config
```
**Why?** Config-driven design allows tuning without code changes. Thresholds must live in `config.yaml`.

---

### 2. **FORBIDDEN: Python Loops Over Keypoint Arrays**
```python
# ❌ WRONG (in velocity_detector or temporal_pose_filter)
for i in range(len(keypoints)):
    distance += np.sqrt((keypoints[i][0] - prev[i][0])**2 + ...)

# ✅ CORRECT (fully vectorized)
distances = np.linalg.norm(keypoints - prev_keypoints, axis=1)  # Shape (17,)
mean_velocity = np.mean(distances)
```
**Why?** Edge devices have constrained CPU. Vectorized NumPy is 100-1000x faster. Use `temporal_pose_filter.py` as reference for vectorization patterns.

---

### 3. **FORBIDDEN: Keeping Video Frame Buffers in Memory**
```python
# ❌ WRONG
frames_buffer = []
for frame in video_source:
    frames_buffer.append(frame)  # Memory leak!

# ✅ CORRECT
for frame in video_source:
    result = detector.detect(frame)
    # Frame auto-released after loop iteration
```
**Why?** On ARM/edge devices, unbuffered frames cause Out-Of-Memory crashes. Process streaming, release immediately. Use `torch.cuda.empty_cache()` after GPU inference.

---

### 4. **FORBIDDEN: Direct Plugin Imports**
```python
# ❌ WRONG
from Atividades_Suspeitas.velocity_detector import VelocityDetector
detector = VelocityDetector(threshold=200.0)  # Breaks modularity!

# ✅ CORRECT
from pipeline.activity_loader import load_activities
activities = load_activities(config.activity_specs())  # Via factory
```
**Why?** Plugin pattern enforces loose coupling. Direct imports prevent dynamic plugin swapping and config-driven behavior.

---

### 5. **FORBIDDEN: Modifying Core Pipeline Files Without Permission**
```
🔴 RESTRICTED FILES (Architectural Core):
  - pipeline/orchestrator.py     → Main event loop
  - Detecao/detector_factory.py  → Backend selection
  - pipeline/config.py           → Config validation

🟡 CAREFUL (Limited changes):
  - main.py                      → CLI parsing only

✅ EDITABLE (Feature extension):
  - Atividades_Suspeitas/*.py    → Add new activity detectors
  - Alertas/*.py                 → Add new alert handlers
  - config.yaml                  → Tune all parameters freely
```
**Why?** Core modules are architectural anchors. Extend via plugins, not modification.

---

### 6. **FORBIDDEN: Detector Instantiation Outside Factory**
```python
# ❌ WRONG
from Detecao.onnx_detector_impl import ONNXDetectorImpl
detector = ONNXDetectorImpl(model_path="./models/end2end.onnx")

# ✅ CORRECT
from Detecao.detector_factory import create_detector
detector = create_detector(config.detector_config())
```
**Why?** Factory pattern allows runtime backend swapping (MMPose ↔ ONNX) without code changes.

---

### 7. **FORBIDDEN: Loading Entire Event Logs Into Memory**
```python
# ❌ WRONG (unbounded memory)
def load_all_events(log_file: str):
    with open(log_file) as f:
        return [json.loads(line) for line in f]

# ✅ CORRECT (paginated)
def load_events_paginated(log_file: str, page: int = 1, page_size: int = 100):
    with open(log_file) as f:
        lines = f.readlines()
        start = (page - 1) * page_size
        return [json.loads(line) for line in lines[start:start+page_size]]
```
**Why?** Alert logs grow unbounded. Always paginate (default 100 events/page). Cap queries to last 7 days.

---

### 8. **FORBIDDEN: GPU Memory Not Released**
```python
# ❌ WRONG
with torch.no_grad():
    output = model(frame_tensor)
# GPU memory lingering...

# ✅ CORRECT
with torch.no_grad():
    output = model(frame_tensor)
torch.cuda.empty_cache()  # Explicit cleanup
```
**Why?** DirectML/CUDA memory fragmentation on edge devices causes OOM. Always cleanup after inference.

---

### 9. **FORBIDDEN: Modifying Model Weights at Runtime**
```
Models (READ-ONLY):
  - ./models/end2end.onnx          → ONNX RTMPose model (~50MB)
  - ./models/pipeline.json         → Reference only
  - ./models/deploy.json           → Metadata only
```
**Why?** Models are production artifacts. Update only via official retraining pipeline. Version with commit hashes.

---

### 10. **FORBIDDEN: Skipping Type Hints**
```python
# ❌ WRONG
def detect_velocity(keypoints, prev_keypoints):
    return distance

# ✅ CORRECT
def detect_velocity(keypoints: np.ndarray, prev_keypoints: Optional[np.ndarray]) -> float:
    """Calculate velocity in pixels/frame."""
    if prev_keypoints is None:
        return 0.0
    distances = np.linalg.norm(keypoints - prev_keypoints, axis=1)
    return float(np.mean(distances))
```
**Why?** Type hints enable early failure detection. IDE catches errors at dev time, not runtime. Use `dataclass`, `Protocol`, `ABC` strictly.

---

## ✅ MANDATORY PATTERNS (Best Practices)

### **1. Always Use Custom Exceptions**
```python
# pipeline/config.py (already defined)
class ConfigError(ValueError):
    """Invalid YAML configuration."""

# pipeline/plugins.py (already defined)
class PluginError(RuntimeError):
    """Plugin load/instantiation failure."""

# Usage: Fail fast at initialization
try:
    config = AppConfig.from_file(args.config)
    config.validate()
except ConfigError as e:
    print(f"[CONFIG] {e}")
    sys.exit(2)
```
**Why?** Custom exceptions clarify error semantics. Detection loop continues on plugin exceptions; config errors abort.

---

### **2. Always Validate at Initialization, Not Runtime**
```python
# ✅ CORRECT
class AntiTheftOrchestrator:
    def __init__(self, config):
        self.detector = create_detector(config.detector_config())  # Fails fast
        self.activities = load_activities(config.activity_specs())  # Validates plugins
        # All expensive operations (model loading) happen here
    
    def run(self):
        # Loop assumes valid state
        for frame in self.video_source:
            # Fast path—no validation
            result = self.detector.detect(frame)
```
**Why?** Move validation cost to startup, not frame-processing loop. Jitter reduction.

---

### **3. Vectorize Everything – Temporal Filter Reference**
```python
# Reference implementation: pipeline/temporal_pose_filter.py
# ✅ PATTERN: All operations are vectorized NumPy

def filter_pose(self, keypoints: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shape: (17, 2) input → (17, 2) smoothed output. ZERO Python loops."""
    
    # Vectorized velocity computation
    velocity = np.linalg.norm(position - prev_position, axis=1)  # Shape (17,)
    
    # Vectorized adaptive smoothing based on velocity magnitude
    is_rapid = velocity > self.config.rapid_movement_threshold  # Shape (17,) bool
    alpha = np.where(is_rapid, self.config.smoothing_factor_fast, self.config.smoothing_factor)
    
    # Vectorized EMA update
    position = alpha[:, None] * keypoints + (1 - alpha[:, None]) * prev_position
    
    # Vectorized occlusion handling
    occluded = scores < self.config.occlusion_confidence_threshold
    position[occluded] = prev_position[occluded] + velocity[occluded] * self.config.velocity_damping
    
    return position, velocity, scores
```
**Runtime:** <0.04ms per frame (fully vectorized). Reference for all temporal/spatial computations.

---

### **4. Protocol-Based Plugin Contracts**
```python
# ✅ All activity detectors must implement:
class BaseActivity(ABC):
    @abstractmethod
    def detecta(self, keypoints: List[tuple], scores: List[float], frame_id: int, timestamp: float) -> Optional[SuspiciousEvent]:
        """Detect suspicious activity. Return SuspiciousEvent or None."""

# ✅ All alert handlers must implement:
class AlertHandler(Protocol):
    def registra_evento(self, event: SuspiciousEvent) -> None: ...

# Validation at load time:
if not hasattr(activity, "detecta"):
    raise TypeError(f"Plugin missing detecta() method: {spec.get('plugin')}")
```
**Why?** Protocol validation fails fast at plugin load, not detection time.

---

### **5. Config Path Resolution**
```python
# ✅ ALWAYS resolve file paths before use
model_path_resolved = Path(model_path).expanduser().resolve()
if not model_path_resolved.exists():
    raise FileNotFoundError(f"Model not found: {model_path_resolved}")
```
**Why?** Relative paths are ambiguous. Always expand `~` (home) and resolve to absolute.

---

## 🔧 Detector Backend Rules (Factory Pattern)

```yaml
# config.yaml
detector:
  type: onnx  # or 'mmpose'
  onnx:
    model_path: ./models/end2end.onnx
    use_gpu: true  # DirectML on Windows, CUDA on Linux
  mmpose:
    model_name: rtmpose-m_8xb256-420e_coco-256x192
    device: cpu
```

**Runtime Switching:**
```bash
python main.py --backend onnx      # Use ONNX with GPU
python main.py --backend mmpose    # Use MMPose on CPU
```

**Implementation Rules:**
- ❌ Do NOT add new backends by modifying `detector_factory.py` directly.
- ✅ Extend factory with new `elif backend_type == 'my_backend'` ONLY for well-justified reasons.
- ✅ All detector impls must implement `detect(frame) -> (keypoints, scores)` contract.

---

## 🎯 Activity Detector Rules (Plugin Pattern)

**Anatomy of a detector:**
```python
from Atividades_Suspeitas.base_activity import BaseActivity, SuspiciousEvent

class MyActivityDetector(BaseActivity):
    def __init__(self, param1: float, param2: int):
        super().__init__("my_activity", threshold=0.5)
        self.param1 = param1
        self.param2 = param2
    
    def detecta(self, keypoints: List[tuple], scores: List[float], 
                frame_id: int, timestamp: float) -> Optional[SuspiciousEvent]:
        """VECTORIZE if possible. Return SuspiciousEvent or None."""
        # Check keypoint validity
        if not keypoints or len(keypoints) < 17:
            return None
        
        # Vectorized computation
        valid_indices = [i for i in range(len(scores)) if scores[i] > 0.3]
        if not valid_indices:
            return None
        
        # Decision logic
        if condition_met:
            return SuspiciousEvent(
                tipo="my_activity",
                timestamp=timestamp,
                confianca=confidence_score,
                frame_id=frame_id,
                descricao="Human-readable description",
                dados_adicionais={"key": value}
            )
        return None
```

**Registration (config.yaml only):**
```yaml
activities:
  - enabled: true
    plugin: Atividades_Suspeitas.my_detector:MyActivityDetector
    params:
      param1: 200.0
      param2: 5
```

**Why this pattern?** Config-driven enables/disables detectors without code changes.

---

## 📢 Alert Handler Rules (Strategy Pattern)

**Contract:**
```python
class AlertSystem:
    def registra_evento(self, evento: SuspiciousEvent) -> None:
        """Handle a suspicious event (log, send, etc.)."""
        # Implementation
```

**Registration (config.yaml):**
```yaml
alerts:
  handlers:
    - enabled: true
      plugin: Alertas.alert_system:AlertSystem
      params:
        pasta_alertas: ./alertas
        save_json: true
        verbose: true
```

**Why?** Multiple handlers can be chained. Each handles independently.

---

## 🎬 Main Loop Constraints

**frame_skip is MANDATORY:**
```python
# config.yaml
runtime:
  frame_skip: 2  # Process every 2nd frame (30fps → 15fps detector)

# In orchestrator (already implemented)
for frame_num, frame in enumerate(video_source):
    if frame_num % config.runtime.frame_skip != 0:
        continue  # Skip frame
    
    # Process
```
**Why?** Real-time constraint. 30fps input → 15fps detection avoids CPU bottleneck.

---

## 🔐 Type Safety Enforcement

**Dataclasses (Config & Events):**
```python
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class SuspiciousEvent:
    tipo: str
    timestamp: float
    confianca: float
    frame_id: int
    pessoa_id: Optional[int] = None
    descricao: str = ""
    dados_adicionais: dict = None
```

**Protocols (Contracts):**
```python
from typing import Protocol

class ActivityDetector(Protocol):
    def detecta(self, keypoints: List[tuple], scores: List[float], 
                frame_id: int, timestamp: float) -> Optional[SuspiciousEvent]: ...
```

**Always add type hints:**
```python
def calculate_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Compute Euclidean distance between two points."""
    return float(np.linalg.norm(p1 - p2))
```

---

## 📊 Performance Guardrails

| Metric | Target | Constraint |
|--------|--------|-----------|
| **Frame processing latency** | <33ms @ 30fps | Vectorize all operations |
| **Temporal filter overhead** | <0.04ms/frame | Use NumPy only (reference: temporal_pose_filter.py) |
| **GPU memory (ONNX)** | <500MB | Call `torch.cuda.empty_cache()` after inference |
| **Event log pagination** | 100 events/page | Cap queries to last 7 days |
| **Plugin load time** | <1s | Fail fast with clear ConfigError/PluginError |

---

## 🚀 Quick Commands

```bash
# Start with default config
python main.py

# Switch to MMPose backend
python main.py --backend mmpose

# Enable GPU (DirectML on Windows, CUDA on Linux)
python main.py --backend onnx --gpu

# Force CPU
python main.py --backend onnx --no-gpu

# Custom config
python main.py --config custom.yaml

# Debug mode
python main.py --debug
```

---

## 📝 File Modification Matrix

| File | Status | When | Notes |
|------|--------|------|-------|
| `config.yaml` | ✅ OPEN | Anytime | Tune detectors, activities, alerts freely |
| `main.py` | 🟡 LIMITED | Rarely | CLI parsing only; no core logic changes |
| `pipeline/orchestrator.py` | 🔴 LOCKED | Architecture review | Main loop is core; respect invariants |
| `Detecao/detector_factory.py` | 🔴 LOCKED | Architecture review | Only extend for new backends with justification |
| `Detecao/*.py (impls)` | 🟡 LIMITED | Bug fixes | No API changes; respect `detect()` contract |
| `Atividades_Suspeitas/*.py` | ✅ OPEN | Always | Add new activity detectors freely |
| `Alertas/*.py` | ✅ OPEN | Always | Add new alert handlers freely |
| `pipeline/temporal_pose_filter.py` | 🟡 LIMITED | Tuning only | Vectorized; no loop-based changes |
| `pipeline/spatial_normalizer.py` | 🟡 LIMITED | Tuning only | Vectorized; no loop-based changes |
| `models/*.onnx` | 🔴 READ-ONLY | Retraining pipeline | Versioned artifacts |

---

## 🧬 Code Style (Must Follow)

```python
# Class Names: PascalCase
class VelocityDetector(BaseActivity): ...

# Methods/Functions: snake_case
def detect_velocity(...) -> float: ...

# Constants: UPPER_SNAKE_CASE
FRAME_SKIP_DEFAULT = 2
MAX_VELOCITY_THRESHOLD = 300.0

# Private methods: _leading_underscore
def _compute_centroid(...) -> np.ndarray: ...

# Config keys: lowercase_snake_case
velocidade_maxima: 200.0
agachamento_threshold: 0.6

# Event log files: ISO 8601
eventos_20260419_145230.json

# Always full type hints:
def my_func(x: np.ndarray, y: Optional[int] = None) -> Dict[str, float]:
    pass
```

---

## 🎯 Decision Tree for New Features

```
New Feature Request?
├─ Need new activity detector?
│  ├─ Create: Atividades_Suspeitas/my_detector.py
│  ├─ Inherit: BaseActivity
│  ├─ Register: config.yaml (activities section)
│  └─ Done (no core changes)
│
├─ Need new alert handler?
│  ├─ Create: Alertas/my_handler.py
│  ├─ Protocol: registra_evento()
│  ├─ Register: config.yaml (alerts section)
│  └─ Done (no core changes)
│
├─ Need new detector backend?
│  ├─ Create: Detecao/my_detector_impl.py
│  ├─ Extend: detector_factory.py (with justification)
│  ├─ Add config: detector_config() support
│  └─ Requires review
│
├─ Need core loop change?
│  ├─ ❌ STOP—Requires architecture review
│  └─ Reach out (high-impact change)
│
└─ Need to tune parameters?
   └─ ✅ Edit config.yaml (no code changes)
```

---

## ⚡ Performance Tuning Checklist

- [ ] **GPU Enabled?** `onnxruntime.get_available_providers()` shows `DmlExecutionProvider` (Windows) or `CUDAExecutionProvider` (Linux)
- [ ] **Frame Skipping Tuned?** Adjust `runtime.frame_skip` (default 2 = 15fps detection)
- [ ] **Model Size OK?** Using RTMPose-M (lightweight) vs RTMPose-L (slower)
- [ ] **Temporal Filter Active?** `temporal_filtering.enabled: true` reduces jitter
- [ ] **Spatial Normalization?** `data.spatial_normalization.enabled: true` for scale-invariant detection
- [ ] **GPU Memory Cleaned?** `torch.cuda.empty_cache()` after ONNX inference
- [ ] **Memory Profiled?** `metrics.memory_usage()` tracked in orchestrator
- [ ] **Event Logs Paginated?** No unbounded reads; use pagination

---

## 📚 Architecture References

- **Main orchestrator:** `pipeline/orchestrator.py`
- **Config validation:** `pipeline/config.py`
- **Plugin loader:** `pipeline/plugins.py` (load_symbol, instantiate_from_spec)
- **Temporal filtering:** `pipeline/temporal_pose_filter.py` (vectorization reference)
- **Spatial normalization:** `pipeline/spatial_normalizer.py` (vectorization reference)
- **Detector factory:** `Detecao/detector_factory.py` (backend selection)
- **Event serialization:** `Alertas/alert_system.py` (JSON schema)
- **Activity base:** `Atividades_Suspeitas/base_activity.py` (plugin contract)

---

**Version:** 2.0 | **Last Updated:** April 19, 2026 | **Status:** LOCKED FOR PRODUCTION
