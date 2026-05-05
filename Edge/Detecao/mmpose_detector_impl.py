"""
Implementação de detecção com MMPose.
"""

import sys
from importlib.machinery import ModuleSpec
from types import ModuleType
from typing import Any

import torch
import torchvision


class _SimpleBypass:
    def __init__(self, name="Bypass"): 
        self.__name__ = name
        self.__path__ = []
        self.__spec__ = ModuleSpec(name=name, loader=None)
        
    def __getattr__(self, name): 
        return self
    def __call__(self, *args, **kwargs): 
        return self


def _custom_nms(boxes, scores, iou_threshold, offset=0, score_threshold=0, max_num=-1):
    keep = torchvision.ops.nms(boxes.cpu(), scores.cpu(), float(iou_threshold))
    if max_num > 0:
        keep = keep[:max_num]
    dets = torch.cat(
        [boxes[keep.to(boxes.device)], scores[keep.to(boxes.device)].unsqueeze(-1)], dim=-1
    )
    return dets, keep.numpy()


def _ensure_mmpose_bypass():
    blacklist = [
        'mmcv._ext',
        'mmcv.ops',
        'mmcv.ops.roi_align',
        'mmcv.ops.nms',
        'mmcv.ops.carafe',
        'mmcv.ops.modulated_deform_conv',
        'mmcv.ops.multi_scale_deform_attn',
        'mmcv.ops.merge_cells',
        'mmcv.ops.active_rotated_filter',
        'mmcv.ops.deform_conv',
    ]

    #for module_name in blacklist:
    #    mock = _SimpleBypass(name=module_name)
    #    setattr(mock, 'nms', _custom_nms)
    #    sys.modules[module_name] = mock

    for m in blacklist:
        mock_module = _SimpleBypass(name=m)
        #mock_module.batched_nms = custom_batched_nms
        mock_module.nms = _custom_nms
        mock_module.roi_align = torchvision.ops.roi_align
        mock_module.RoIAlign = torchvision.ops.RoIAlign
        sys.modules[m] = mock_module



class MMPoseDetectorImpl:
    """Usa MMPoseInferencer (library-based)."""
    
    def __init__(self, model_name='rtmpose-m_8xb256-420e_coco-256x192', device='cpu'):
        """
        Inicializa detector MMPose.
        
        Args:
            model_name: nome do modelo MMPose (ex: rtmpose-m_8xb256-420e_coco-256x192)
            device: 'cpu', 'cuda', etc
        """
        self.model_name = model_name
        self.device = device
        self._inferencer: Any = None
        
        print(f"  [MMPose] Carregando {model_name}...")
        _ensure_mmpose_bypass()

        from mmpose.apis import MMPoseInferencer

        self._inferencer = MMPoseInferencer(
            pose2d=model_name,
            det_model='whole_image',
            device=device
        )
        print(f"  [OK] MMPose pronto")
    
    def detect(self, frame):
        """Detecta pose com MMPose."""
        try:
            if self._inferencer is None:
                return [], []

            with torch.no_grad():
                results_obj = self._inferencer(frame, return_vis=True, show=False)

                if hasattr(results_obj, '__next__'):
                    result = next(results_obj)
                else:
                    result = results_obj
            
            keypoints = []
            scores = []
            
            # Parse resultado
            if 'predictions' in result and len(result['predictions']) > 0:
                preds = result['predictions'][0]
                
                if isinstance(preds, dict):
                    people = [preds]
                elif isinstance(preds, list):
                    people = preds
                else:
                    people = []
                
                for person in people:
                    kps = person.get('keypoints', [])
                    scr = person.get('keypoint_scores', [])
                    
                    if len(kps) > 0:
                        # Converte para listas Python limpas
                        if hasattr(kps, 'tolist'):
                            keypoints = kps.tolist()
                        else:
                            keypoints = list(kps)
                        
                        if len(scr) > 0:
                            if hasattr(scr, 'tolist'):
                                scores = scr.tolist()
                            else:
                                scores = list(scr)
                        else:
                            scores = [1.0] * len(keypoints)
                        break
            
            return keypoints, scores
        
        except Exception as e:
            print(f"[ERRO] MMPose detect: {e}")
            return [], []
    
    def get_info(self):
        """Retorna info do detector."""
        return {
            'backend': 'MMPose',
            'model': self.model_name,
            'device': self.device,
            'type': 'library'
        }
