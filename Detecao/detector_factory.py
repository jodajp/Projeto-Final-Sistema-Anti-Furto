"""
Factory para criar detector de pose apropriado.
Suporta MMPose (library) e ONNX (WinML/DirectML).
"""

from pathlib import Path
from typing import Any, Dict


def create_detector(backend_config: Dict[str, Any]) -> Any:
    """
    Cria detector baseado em configuração.
    
    Args:
        backend_config: dict com chaves:
            - type: 'mmpose' ou 'onnx'
            - model_name: (se mmpose) nome do modelo
            - model_path: (se onnx) caminho ao ficheiro .onnx
            - device: (se mmpose) 'cpu', 'cuda', etc
            - use_gpu: (se onnx) True/False
    
    Returns:
        PoseDetector: detector inicializado
    """
    backend_type = backend_config.get('type', 'mmpose').lower()
    
    if backend_type == 'mmpose':
        from .mmpose_detector_impl import MMPoseDetectorImpl
        model_name = (
            backend_config.get('model_name')
            or backend_config.get('model')
            or backend_config.get('modelo')
            or 'rtmpose-m_8xb256-420e_coco-256x192'
        )
        device = backend_config.get('device', 'cpu')
        return MMPoseDetectorImpl(model_name=model_name, device=device)
    
    elif backend_type == 'onnx':
        from .onnx_detector_impl import ONNXDetectorImpl
        model_path = backend_config.get('model_path')
        
        if not model_path:
            raise ValueError("[ERRO] ONNX backend requer 'model_path'")
        
        model_path_resolved = Path(model_path).expanduser().resolve()
        if not model_path_resolved.exists():
            raise FileNotFoundError(f"[ERRO] Modelo ONNX não encontrado: {model_path_resolved}")
        
        use_gpu = backend_config.get('use_gpu', True)
        return ONNXDetectorImpl(model_path=str(model_path_resolved), use_gpu=use_gpu)
    
    else:
        raise ValueError(f"[ERRO] Backend desconhecido: {backend_type}")
