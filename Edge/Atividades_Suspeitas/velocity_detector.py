"""
Detector de Velocidade Anormal
Detecta movimentos muito rápidos (típicos de roubos) com normalização de escala
"""

import sys
from pathlib import Path
from typing import List, Optional, Dict
import numpy as np

# Adiciona diretório Edge ao path para resolver imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from .base_activity import BaseActivity, SuspiciousEvent
from pipeline.spatial_normalizer import NormalizedPose


class VelocityDetector(BaseActivity):
    """Detecta velocidade de movimento anormal com independência de escala."""
    
    def __init__(self, velocidade_maxima: float = 200.0, cooldown_frames: int = 60):
        super().__init__("velocidade", threshold=0.6)
        self.velocidade_maxima = velocidade_maxima
        self.cooldown_frames = cooldown_frames
        
        # Histórico indexado por track_id
        self.ultima_posicao = {}  # track_id -> pelvis (x, y)
        self.ultimo_frame_id = {}  # track_id -> frame_id
        self.frames_since_last_alerts = {}  # track_id -> int
        
    def limpa_tracks_inativas(self, ids_presentes: set):
        """Limpa o histórico de tracks inativas para evitar vazamento de memória."""
        for track_id in list(self.ultima_posicao.keys()):
            if track_id not in ids_presentes:
                self.ultima_posicao.pop(track_id, None)
                self.ultimo_frame_id.pop(track_id, None)
                self.frames_since_last_alerts.pop(track_id, None)
                
    def detecta(self, 
                norm_pose: NormalizedPose,
                frame_id: int,
                timestamp: float,
                track_id: Optional[int] = None) -> Optional[SuspiciousEvent]:
        tid = 0 if track_id is None else track_id
        
        if tid not in self.frames_since_last_alerts:
            self.frames_since_last_alerts[tid] = self.cooldown_frames
            
        self.frames_since_last_alerts[tid] += 1
        
        if not norm_pose or not norm_pose.is_valid:
            return None
            
        # Posição absoluta do pelvis na imagem (para medir locomoção)
        pelvis_atual = norm_pose.pelvis
        torso_length = norm_pose.torso_length
        
        if torso_length <= 0:
            return None
            
        # Se é o primeiro frame da track, guarda posição e retorna
        if tid not in self.ultima_posicao:
            self.ultima_posicao[tid] = pelvis_atual
            self.ultimo_frame_id[tid] = frame_id
            return None
            
        # Calcula deslocamento absoluto
        diff = pelvis_atual - self.ultima_posicao[tid]
        distancia_raw = np.linalg.norm(diff)
        
        frames_decorridos = frame_id - self.ultimo_frame_id[tid]
        if frames_decorridos > 0:
            velocidade_raw = distancia_raw / frames_decorridos
        else:
            velocidade_raw = 0.0
            
        # Normaliza a velocidade com base no tamanho do torso (em relação a uma referência de 100px)
        velocidade = velocidade_raw * (100.0 / torso_length)
        
        # Atualiza histórico da track
        self.ultima_posicao[tid] = pelvis_atual
        self.ultimo_frame_id[tid] = frame_id
        
        # Só emite alerta após expirar o cooldown
        if velocidade > self.velocidade_maxima and self.frames_since_last_alerts[tid] >= self.cooldown_frames:
            self.frames_since_last_alerts[tid] = 0
            
            evento = SuspiciousEvent(
                tipo="velocidade",
                timestamp=timestamp,
                confianca=min(velocidade / (self.velocidade_maxima * 2), 1.0),
                frame_id=frame_id,
                pessoa_id=track_id,
                descricao=f"Velocidade anormal: {velocidade:.1f} px/frame (normalizada)",
                dados_adicionais={
                    'velocidade_normalizada': float(velocidade),
                    'velocidade_raw': float(velocidade_raw),
                    'torso_length': float(torso_length),
                    'threshold': float(self.velocidade_maxima)
                }
            )
            self.registra_evento(evento)
            return evento
            
        return None
