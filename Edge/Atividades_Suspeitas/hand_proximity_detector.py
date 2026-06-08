"""
Detector de Proximidade das Mãos (Ocultação de Produtos)
Detecta quando as mãos (pulsos) se aproximam ou permanecem na zona dos bolsos/cintura com independência de escala.
"""

import sys
from pathlib import Path
from typing import List, Optional
import numpy as np

# Adiciona diretório Edge ao path para resolver imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from .base_activity import BaseActivity, SuspiciousEvent
from Detecao.skeleton import LEFT_WRIST, RIGHT_WRIST
from pipeline.spatial_normalizer import NormalizedPose


class HandProximityDetector(BaseActivity):
    """Detecta ocultação de produtos analisando a distância das mãos ao quadril."""
    
    def __init__(self, distancia_maxima: float = 150.0, tempo_minimo: int = 10, cooldown_frames: int = 60):
        super().__init__("ocultacao_produto", threshold=0.5)
        self.distancia_maxima = distancia_maxima
        self.tempo_minimo = tempo_minimo
        self.cooldown_frames = cooldown_frames
        
        # Históricos por track_id
        self.frames_em_risco = {}  # track_id -> frames
        self.frames_since_last_alerts = {}  # track_id -> int
        
    def limpa_tracks_inativas(self, ids_presentes: set):
        """Limpa o histórico de tracks inativas para evitar vazamento de memória."""
        for track_id in list(self.frames_em_risco.keys()):
            if track_id not in ids_presentes:
                self.frames_em_risco.pop(track_id, None)
                self.frames_since_last_alerts.pop(track_id, None)
                
    def detecta(self, 
                norm_pose: NormalizedPose,
                frame_id: int,
                timestamp: float,
                track_id: Optional[int] = None) -> Optional[SuspiciousEvent]:
        tid = 0 if track_id is None else track_id
        
        if tid not in self.frames_em_risco:
            self.frames_em_risco[tid] = 0
            self.frames_since_last_alerts[tid] = self.cooldown_frames
            
        self.frames_since_last_alerts[tid] += 1
        
        if not norm_pose or not norm_pose.is_valid:
            self.frames_em_risco[tid] = 0
            return None
            
        sc = norm_pose.scores
        kp_norm = norm_pose.keypoints  # Keypoints já normalizados e centrados no pelvis (0,0)
        
        # Filtra os pulsos (mãos) válidos e calcula distâncias ao pelvis (0,0) de forma simples
        distances = []
        if sc[LEFT_WRIST] > 0.3:
            distances.append(np.linalg.norm(kp_norm[LEFT_WRIST]))
        if sc[RIGHT_WRIST] > 0.3:
            distances.append(np.linalg.norm(kp_norm[RIGHT_WRIST]))
            
        if len(distances) == 0:
            self.frames_em_risco[tid] = 0
            return None
            
        min_dist_norm = min(distances)
        
        # Converte a distância normalizada de volta a pixels relativos (torso padrão = 100px)
        dist_scaled_px = min_dist_norm * 100.0
        
        # Calcula o risco em escala 0.0 a 1.0
        risk = max(0.0, 1.0 - (dist_scaled_px / self.distancia_maxima))
        
        if risk >= self.threshold:
            self.frames_em_risco[tid] += 1
            
            if self.frames_em_risco[tid] >= self.tempo_minimo and self.frames_since_last_alerts[tid] >= self.cooldown_frames:
                self.frames_since_last_alerts[tid] = 0
                
                evento = SuspiciousEvent(
                    tipo=self.nome,
                    timestamp=timestamp,
                    confianca=float(risk),
                    frame_id=frame_id,
                    pessoa_id=track_id,
                    descricao=f"Possível ocultação: Mão próxima aos bolsos (Risco: {risk*100:.1f}%)",
                    dados_adicionais={
                        'risco_ocultacao': float(risk),
                        'distancia_px_normalizada': float(dist_scaled_px),
                        'frames_consecutivos': int(self.frames_em_risco[tid])
                    }
                )
                self.registra_evento(evento)
                return evento
        else:
            self.frames_em_risco[tid] = 0
            
        return None