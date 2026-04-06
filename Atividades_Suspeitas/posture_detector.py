"""
Detector de Agachamento
Detecta quando pessoa está agachada (típico de roubo em loja)
"""

from typing import List, Optional
from .base_activity import BaseActivity, SuspiciousEvent
import numpy as np

# Índices de keypoints para o corpo (COCO 17)
NOSE = 0
NECK = 1  # Média dos ombros
OMBRO_ESQ = 5
OMBRO_DIR = 6
QUADRIL_ESQ = 11
QUADRIL_DIR = 12
TORNOZELO_ESQ = 15
TORNOZELO_DIR = 16

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
        keypoints_importantes = [OMBRO_ESQ, OMBRO_DIR, QUADRIL_ESQ, QUADRIL_DIR, TORNOZELO_ESQ, TORNOZELO_DIR]
        
        confiancas_validas = all(
            i < len(scores) and scores[i] > 0.3 
            for i in keypoints_importantes
        )
        
        if not confiancas_validas:
            self.frames_agachado = 0
            return None
        
        # Calcula alturas
        altura_ombros = (keypoints[OMBRO_ESQ][1] + keypoints[OMBRO_DIR][1]) / 2
        altura_quadris = (keypoints[QUADRIL_ESQ][1] + keypoints[QUADRIL_DIR][1]) / 2
        altura_tornozelos = (keypoints[TORNOZELO_ESQ][1] + keypoints[TORNOZELO_DIR][1]) / 2
        
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
