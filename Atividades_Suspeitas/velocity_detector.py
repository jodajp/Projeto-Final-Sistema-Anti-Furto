"""
Detector de Velocidade Anormal
Detecta movimentos muito rápidos (típicos de roubos)
"""

from typing import List, Optional, Dict
from .base_activity import BaseActivity, SuspiciousEvent
import numpy as np

class VelocityDetector(BaseActivity):
    """Detecta velocidade de movimento anormal."""
    
    def __init__(self, velocidade_maxima: float = 200.0):
        super().__init__("velocidade", threshold=0.6)
        self.velocidade_maxima = velocidade_maxima
        
        # Histórico de última posição
        self.ultima_posicao = None
        self.ultimo_frame_id = None
        
    def detecta(self, 
                keypoints: List[tuple], 
                scores: List[float],
                frame_id: int,
                timestamp: float) -> Optional[SuspiciousEvent]:
        """
        Detecta velocidade anormal.
        
        Calcula a distância do centroide entre frames sucessivos
        e compara com o threshold.
        """
        
        if not keypoints or len(keypoints) == 0:
            return None
        
        # Calcula centroide apenas com keypoints confiáveis
        pontos_validos = []
        for i, (x, y) in enumerate(keypoints):
            if i < len(scores) and scores[i] > 0.3:
                pontos_validos.append((x, y))
        
        if not pontos_validos:
            self.ultima_posicao = None
            return None
        
        # Centroide atual
        centroide_atual = (
            np.mean([p[0] for p in pontos_validos]),
            np.mean([p[1] for p in pontos_validos])
        )
        
        # Se é primeira vez, apenas guarda posição
        if self.ultima_posicao is None:
            self.ultima_posicao = centroide_atual
            self.ultimo_frame_id = frame_id
            return None
        
        # Calcula distância e velocidade
        dx = centroide_atual[0] - self.ultima_posicao[0]
        dy = centroide_atual[1] - self.ultima_posicao[1]
        distancia = np.sqrt(dx**2 + dy**2)
        
        frames_decorridos = frame_id - self.ultimo_frame_id
        if frames_decorridos > 0:
            velocidade = distancia / frames_decorridos
        else:
            velocidade = 0
        
        # Atualiza histórico
        self.ultima_posicao = centroide_atual
        self.ultimo_frame_id = frame_id
        
        # Detecta velocidade anormal
        if velocidade > self.velocidade_maxima:
            evento = SuspiciousEvent(
                tipo="velocidade",
                timestamp=timestamp,
                confianca=min(velocidade / (self.velocidade_maxima * 2), 1.0),
                frame_id=frame_id,
                descricao=f"Velocidade anormal detectada: {velocidade:.1f} px/frame",
                dados_adicionais={
                    'velocidade': velocidade,
                    'threshold': self.velocidade_maxima,
                    'centroide': centroide_atual
                }
            )
            self.registra_evento(evento)
            return evento
        
        return None
