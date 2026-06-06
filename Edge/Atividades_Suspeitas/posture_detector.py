"""
Detector de Agachamento
Detecta quando pessoa está agachada (típico de roubo em loja)
"""

from typing import List, Optional
from .base_activity import BaseActivity, SuspiciousEvent
import numpy as np
from Detecao.skeleton import NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE

# NECK não é um keypoint padrão do COCO 17; é computado a partir dos ombros.

class PostureDetector(BaseActivity):
    """Detecta postura (agachamento, deitado, etc)."""
    
    def __init__(self, agachamento_threshold: float = 0.6, tempo_minimo: int = 5):
        super().__init__("agachamento", threshold=0.5)
        self.agachamento_threshold = agachamento_threshold
        self.tempo_minimo = tempo_minimo
        
        # Contador de frames em agachamento
        self.frames_agachado = 0
        
    def detecta(self, 
                keypoints: List[tuple], 
                scores: List[float],
                frame_id: int,
                timestamp: float) -> Optional[SuspiciousEvent]:
        """
        Detecta agachamento.
        
        Compara altura do quadril com altura do corpo total.
        Se quadril está muito baixo = agachado.
        """
        
        if not keypoints or len(keypoints) < 17:
            self.frames_agachado = 0
            return None
        
        # Verifica confiança dos keypoints importantes
        keypoints_importantes = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE]
        
        confiancas_validas = all(
            i < len(scores) and scores[i] > 0.3 
            for i in keypoints_importantes
        )
        
        if not confiancas_validas:
            self.frames_agachado = 0
            return None
        
        # Calcula alturas
        altura_ombros = (keypoints[LEFT_SHOULDER][1] + keypoints[RIGHT_SHOULDER][1]) / 2
        altura_quadris = (keypoints[LEFT_HIP][1] + keypoints[RIGHT_HIP][1]) / 2
        altura_tornozelos = (keypoints[LEFT_ANKLE][1] + keypoints[RIGHT_ANKLE][1]) / 2
        
        # Altura total do corpo
        altura_total = altura_tornozelos - altura_ombros
        
        if altura_total <= 0:
            return None
        
        # Distância do quadril até o pé
        distancia_quadril_pe = altura_tornozelos - altura_quadris
        
        # Razão: se quadril está perto dos pés, está agachado
        razao = distancia_quadril_pe / altura_total
        
        # Se razão < threshold, está agachado
        is_agachado = razao < self.agachamento_threshold
        
        if is_agachado:
            self.frames_agachado += 1
            
            # Alerta apenas após tempo mínimo agachado
            if self.frames_agachado >= self.tempo_minimo:
                evento = SuspiciousEvent(
                    tipo="agachamento",
                    timestamp=timestamp,
                    confianca=min(1.0 - razao, 1.0),  # Quanto mais abaixo, mais confiança
                    frame_id=frame_id,
                    descricao=f"Agachamento detectado por {self.frames_agachado} frames",
                    dados_adicionais={
                        'razao': razao,
                        'threshold': self.agachamento_threshold,
                        'frames_agachado': self.frames_agachado
                    }
                )
                self.registra_evento(evento)
                return evento
        else:
            self.frames_agachado = 0
        
        return None
