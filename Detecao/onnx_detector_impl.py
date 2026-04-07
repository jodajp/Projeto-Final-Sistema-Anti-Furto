"""
ONNX detector using rtmlib with DirectML/WinML GPU support.
Monkey-patches rtmlib to add DirectML backend for AMD/Intel/NVIDIA GPUs.
Combines GPU speed with rtmlib's accurate postprocessing.
Includes temporal smoothing for stable detection across frames.
"""
import numpy as np
import platform
from collections import deque


class ONNXDetectorImpl:
    """ONNX detector using rtmlib with DirectML GPU support"""
    
    def __init__(self, model_path, use_gpu=True):
        """
        Initialize rtmlib RTMO detector with DirectML GPU support.
        
        Args:
            model_path: path to .onnx file
            use_gpu: whether to use GPU (DirectML on Windows, CUDA on Linux)
        """
        print(f"  [ONNX] Carregando {model_path} com rtmlib + GPU...")
        
        try:
            from rtmlib import RTMO
            import rtmlib.tools.base as rtmlib_base
        except ImportError as exc:
            raise ImportError(
                "ONNX backend requer rtmlib. Instale com: pip install rtmlib"
            ) from exc
        
        # Select device based on GPU availability and platform
        device = 'cpu'
        if use_gpu:
            import onnxruntime as ort
            available = ort.get_available_providers()
            
            if platform.system() == 'Windows':
                if 'CUDAExecutionProvider' in available:
                    device = 'cuda'
                    print(f"  [ONNX] Usando CUDA")
            
                elif 'DmlExecutionProvider' in available:
                    device = 'winml'  # Use DirectML on Windows
                    print(f"  [ONNX] Usando WinML/DirectML")
                    print(f"  [ONNX] Adicionando suporte DirectML/WinML ao rtmlib...")
            
                    # Fix para adicionar suporte a DirectML/WinML no rtmlib
                    if 'winml' not in rtmlib_base.RTMLIB_SETTINGS['onnxruntime']:
                        rtmlib_base.RTMLIB_SETTINGS['onnxruntime']['winml'] = 'DmlExecutionProvider'
                        print(f"  [OK] WinML/DirectML configurado em rtmlib")

            else:
                if 'CUDAExecutionProvider' in available:
                    device = 'cuda'
                    print(f"  [ONNX] Usando CUDA")
        
        # Inicializa o modelo com rtmlib, usando GPU se disponível
        try:
            self.model = RTMO(
                onnx_model=model_path,
                backend='onnxruntime',
                device=device,
                model_input_size=(640, 640),
                nms_thr=0.65,      # Increased from 0.45 to reduce overlapping detections
                score_thr=0.4       # Increased from 0.1 to filter weak detections
            )
        except Exception as e:
            if device in ['winml', 'cuda']:
                print(f"  [AVISO] GPU falhou ({device}), voltando para CPU")
                self.model = RTMO(
                    onnx_model=model_path,
                    backend='onnxruntime',
                    device='cpu',
                    model_input_size=(640, 640),
                    nms_thr=0.65,      # Increased to reduce overlapping detections
                    score_thr=0.4       # Increased to filter weak detections
                )
            else:
                print(f"  [ERRO] Falha ao carregar modelo: {e}")
                raise
        
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.device = device
        
        # Temporal smoothing parameters
        self.smoothing_factor = 0.4     # EMA factor (lower = more smoothing, 0-1)
        self.prev_keypoints = None      # Previous frame keypoints for smoothing
        self.prev_scores = None         # Previous frame scores
        self.frame_buffer = deque(maxlen=3)  # Buffer last 3 frames for stability
        self.detection_enabled = True   # Allow disabling detection if too unstable
        
        # Apenas para debug
        if device == 'winml':
            self.provider_info = 'WinML/DirectML (AMD/Intel/NVIDIA)'
        elif device == 'cuda':
            self.provider_info = 'CUDA (NVIDIA)'
        else:
            self.provider_info = 'CPU (rtmlib optimized)'
        
        print(f"  [OK] ONNX pronto com rtmlib (Provider: {self.provider_info})")
        print(f"  [INFO] Temporal smoothing enabled (factor={self.smoothing_factor})")
    
    def detect(self, frame):
        """
        Detect pose keypoints using rtmlib with GPU + temporal smoothing.
        
        Returns: (keypoints_list, scores)
            - keypoints_list: list of [x, y] coordinates (temporally smoothed)
            - scores: confidence scores [0.0-1.0]
        """
        try:
            # rtmlib handles all preprocessing, inference, and postprocessing
            keypoints, scores = self.model(frame)
            
            # keypoints shape = (num_people, 17, 2)
            # scores shape = (num_people, 17)
            
            # Remove low-confidence detections per frame
            valid_mask = scores.mean(axis=1) > 0.3  # Filter persons with avg confidence < 0.3
            keypoints = keypoints[valid_mask]
            scores = scores[valid_mask]
            
            # Nenhuma deteção confiável, retorna última detecção estável
            if len(keypoints) == 0:
                if self.prev_keypoints is not None:
                    return self.prev_keypoints, self.prev_scores
                return [], []
            
            ###################
            # Esta parte esta apenas pronta para lidar com 1 pessoa em vez de múltiplas
            # TODO: Implementar lógica para lidar com múltiplas pessoas (tracking, associação, etc)
            ###################
            
            # Select best person (highest average confidence)
            person_confidences = scores.mean(axis=1)
            best_person_idx = np.argmax(person_confidences)
            best_kpts = keypoints[best_person_idx]      # [17, 2]
            best_scores = scores[best_person_idx]       # [17]
            
            # Apply temporal smoothing (EMA filter)
            if self.prev_keypoints is not None:
                # Exponential Moving Average smoothing
                best_kpts = (
                    self.smoothing_factor * best_kpts + 
                    (1 - self.smoothing_factor) * np.array(self.prev_keypoints)
                )
            
            # Validate stability - check if scale changed drastically
            if self.prev_keypoints is not None:
                prev_scale = self._estimate_scale(self.prev_keypoints)
                curr_scale = self._estimate_scale(best_kpts)
                
                scale_ratio = curr_scale / (prev_scale + 1e-6)
                
                # If scale changed too much, reduce smoothing aggressively
                if not (0.85 < scale_ratio < 1.15):  # Allow ±15% scale change
                    # Use more smoothing to suppress outliers
                    best_kpts = (
                        0.2 * best_kpts + 
                        0.8 * np.array(self.prev_keypoints)
                    )
            
            # Convert to list format for compatibility
            kpts_list = []
            scores_list = []
            
            for idx, kpt in enumerate(best_kpts):
                x, y = float(kpt[0]), float(kpt[1])
                # Clip to frame bounds (safety)
                x = max(0, min(x, frame.shape[1] - 1))
                y = max(0, min(y, frame.shape[0] - 1))
                kpts_list.append([x, y])
                
                # Get confidence score for this keypoint
                conf = float(best_scores[idx]) if idx < len(best_scores) else 1.0
                scores_list.append(conf)
            
            # Store for next frame smoothing
            self.prev_keypoints = kpts_list
            self.prev_scores = scores_list
            
            return kpts_list, scores_list
        
        except Exception as e:
            print(f"[ERRO] ONNX detect: {e}")
            import traceback
            traceback.print_exc()
            # Return previous detection if available
            if self.prev_keypoints is not None:
                return self.prev_keypoints, self.prev_scores
            return [], []
    
    def _estimate_scale(self, keypoints):
        """Estimate skeleton scale from keypoint spread."""
        kpts_array = np.array(keypoints)
        x_range = kpts_array[:, 0].max() - kpts_array[:, 0].min()
        y_range = kpts_array[:, 1].max() - kpts_array[:, 1].min()
        return np.sqrt(x_range * y_range)  # Geometric mean-like measure
    
    def get_info(self):
        """Return detector metadata."""
        return {
            'backend': 'ONNX (rtmlib + WinML/DirectML)',
            'provider': self.provider_info,
            'model': self.model_path,
            'type': 'onnx'
        }
