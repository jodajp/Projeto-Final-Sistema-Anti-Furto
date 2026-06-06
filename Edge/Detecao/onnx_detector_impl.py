"""
ONNX detector using rtmlib with DirectML/WinML GPU support.
Returns all detected people as raw pose batches.
"""
import numpy as np
import platform


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
                if 'DmlExecutionProvider' in available:
                    device = 'winml'  # Use DirectML on Windows
                    print(f"  [ONNX] Usando WinML/DirectML")
                    print(f"  [ONNX] Adicionando suporte DirectML/WinML ao rtmlib...")
            
                    # Fix para adicionar suporte a DirectML/WinML no rtmlib
                    if 'winml' not in rtmlib_base.RTMLIB_SETTINGS['onnxruntime']:
                        rtmlib_base.RTMLIB_SETTINGS['onnxruntime']['winml'] = 'DmlExecutionProvider'
                        print(f"  [OK] WinML/DirectML configurado em rtmlib")

                elif 'CUDAExecutionProvider' in available:
                    device = 'cuda'
                    print(f"  [ONNX] Usando CUDA")
            else:
                if 'CUDAExecutionProvider' in available:
                    device = 'cuda'
                    print(f"  [ONNX] Usando CUDA")
        
        # Import onnxruntime and set up optimized SessionOptions wrapper
        # Otimizações redução de 60ms -> 36ms
        try:
            import onnxruntime as ort
            import os
            
            original_InferenceSession = ort.InferenceSession

            def custom_InferenceSession(path_or_bytes, providers=None, sess_options=None, **kwargs):
                if sess_options is None:
                    sess_options = ort.SessionOptions()
                    # 1. Enable all graph optimizations
                    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    # 2. Force sequential execution mode for DirectML/CUDA and CPU stability
                    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                    
                    # 3. Configure threading to prevent oversubscription
                    cpu_count = os.cpu_count() or 4
                    sess_options.intra_op_num_threads = max(1, cpu_count // 2)
                    sess_options.inter_op_num_threads = 1
                    
                    # 4. DirectML specific optimizations
                    if providers and 'DmlExecutionProvider' in providers:
                        sess_options.enable_mem_pattern = False

                # Register CPUExecutionProvider as fallback for safety
                if providers:
                    if 'DmlExecutionProvider' in providers and 'CPUExecutionProvider' not in providers:
                        providers = list(providers) + ['CPUExecutionProvider']
                    elif 'CUDAExecutionProvider' in providers and 'CPUExecutionProvider' not in providers:
                        providers = list(providers) + ['CPUExecutionProvider']

                print(f"  [ONNX OPT] Criando InferenceSession com SessionOptions otimizadas:")
                print(f"    - Graph Optimization: {sess_options.graph_optimization_level}")
                print(f"    - Execution Mode: {sess_options.execution_mode}")
                print(f"    - Intra OP Threads: {sess_options.intra_op_num_threads}")
                print(f"    - Memory Pattern: {sess_options.enable_mem_pattern}")
                print(f"    - Providers: {providers}")

                return original_InferenceSession(path_or_bytes, providers=providers, sess_options=sess_options, **kwargs)

            ort.InferenceSession = custom_InferenceSession
        except ImportError:
            ort = None

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
        finally:
            # Restore original InferenceSession to avoid side effects elsewhere
            if ort is not None:
                ort.InferenceSession = original_InferenceSession
        
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.device = device
        
        # Apenas para debug
        if device == 'winml':
            self.provider_info = 'WinML/DirectML (AMD/Intel/NVIDIA)'
        elif device == 'cuda':
            self.provider_info = 'CUDA (NVIDIA)'
        else:
            self.provider_info = 'CPU (rtmlib optimized)'
        
        print(f"  [OK] ONNX pronto com rtmlib (Provider: {self.provider_info})")
    
    def detect(self, frame):
        """
        Detect pose keypoints using rtmlib.
        
        Returns: (keypoints_list, scores)
            - keypoints_list: list of people, each one a list of [x, y] coordinates
            - scores: list of people, each one a list of confidence scores [0.0-1.0]
        """
        try:
            # rtmlib handles all preprocessing, inference, and postprocessing
            keypoints, scores = self.model(frame)
            
            # keypoints shape = (num_people, 17, 2)
            # scores shape = (num_people, 17)
            
            # Remove low-confidence detections per frame
            valid_mask = scores.mean(axis=1) > 0.05  # Filter persons with avg confidence < 0.05
            keypoints = keypoints[valid_mask]
            scores = scores[valid_mask]
            
            # Nenhuma deteção confiável
            if len(keypoints) == 0:
                return [], []

            keypoints = np.asarray(keypoints, dtype=np.float32)
            scores = np.asarray(scores, dtype=np.float32)

            clipped_keypoints = np.clip(
                keypoints,
                [0.0, 0.0],
                [frame.shape[1] - 1.0, frame.shape[0] - 1.0],
            )

            return clipped_keypoints.tolist(), scores.tolist()
        
        except Exception as e:
            print(f"[ERRO] ONNX detect: {e}")
            import traceback
            traceback.print_exc()
            return [], []
    
    def get_info(self):
        """Return detector metadata."""
        return {
            'backend': 'ONNX (rtmlib + WinML/DirectML)',
            'provider': self.provider_info,
            'model': self.model_path,
            'type': 'onnx'
        }
