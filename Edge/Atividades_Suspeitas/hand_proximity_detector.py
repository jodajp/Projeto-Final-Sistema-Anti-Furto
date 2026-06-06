"""
Detector de Proximidade das Mãos (Ocultação de Produtos)
Detecta quando as mãos (pulsos) se aproximam ou permanecem na zona dos bolsos/cintura.
"""

from typing import List, Optional
import numpy as np
from .base_activity import BaseActivity, SuspiciousEvent
from Detecao.skeleton import LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP

class HandProximityDetector(BaseActivity):
    """Detecta ocultação de produtos analisando a distância das mãos ao tronco/bolsos."""
    
    def __init__(self, distancia_maxima: float = 150.0, tempo_minimo: int = 10):
        """
        Args:
            distancia_maxima: Distância máxima em píxeis para considerar risco.
            tempo_minimo: Número de frames consecutivos que a mão tem de estar na zona.
        """
        # Threshold base para o "Risco" (0.5 = 50% de risco mínimo exigido)
        super().__init__("ocultacao_produto", threshold=0.5)
        self.distancia_maxima = distancia_maxima
        self.tempo_minimo = tempo_minimo
        self.frames_em_risco = 0
        
    def detecta(self, 
                keypoints: List[tuple], 
                scores: List[float],
                frame_id: int,
                timestamp: float) -> Optional[SuspiciousEvent]:
        
        if not keypoints or len(keypoints) < 17:
            self.frames_em_risco = 0
            return None
            
        # Converter para arrays NumPy para vetorização (Performance Edge)
        kp = np.asarray(keypoints, dtype=np.float32)
        sc = np.asarray(scores, dtype=np.float32)
        
        # Acha os bolsos/cintura (quadris)
        valid_hips = sc[[LEFT_HIP, RIGHT_HIP]] > 0.3
        if not valid_hips.any():
            self.frames_em_risco = 0
            return None
        
        # Calcula o centro geométrico da cintura vetorizadamente
        pocket_center = kp[[LEFT_HIP, RIGHT_HIP]][valid_hips].mean(axis=0)
        
        # Filtrar os pulsos válidos
        valid_wrists = sc[[LEFT_WRIST, RIGHT_WRIST]] > 0.3
        if not valid_wrists.any():
            self.frames_em_risco = 0
            return None
        
        wrists_kp = kp[[LEFT_WRIST, RIGHT_WRIST]][valid_wrists]
        
        # Calcula a distâncias de todos os pulsos válidos ao centro dos bolsos de uma só vez
        distances = np.linalg.norm(wrists_kp - pocket_center, axis=1)
        min_dist = np.min(distances)
        
        # Calcula o Risco (Escala 0.0 a 1.0)
        # Risco é 1.0 se a distância for 0, e 0.0 se a distância for >= distancia_maxima
        risk = max(0.0, 1.0 - (min_dist / self.distancia_maxima))
        
        # Avalia e dispara um evento se mantiver o padrão
        if risk >= self.threshold:
            self.frames_em_risco += 1
            
            if self.frames_em_risco >= self.tempo_minimo:
                evento = SuspiciousEvent(
                    tipo=self.nome,
                    timestamp=timestamp,
                    confianca=float(risk),
                    frame_id=frame_id,
                    descricao=f"Possível ocultação: Mão próxima aos bolsos por {self.frames_em_risco} frames (Risco: {risk*100:.1f}%)",
                    dados_adicionais={
                        'risco_ocultacao': float(risk),
                        'distancia_px': float(min_dist),
                        'frames_consecutivos': self.frames_em_risco,
                    }
                )
                self.registra_evento(evento)
                return evento
        else:
            # Se as mãos se afastarem, faz reset ao contador
            self.frames_em_risco = 0
            
        return None