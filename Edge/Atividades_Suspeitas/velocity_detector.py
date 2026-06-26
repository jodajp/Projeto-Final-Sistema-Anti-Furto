"""
Detector de Velocidade Anormal
Detecta movimentos muito rápidos (típicos de roubos) com normalização de escala e independência de FPS.
"""

from typing import Optional
import numpy as np

from .base_activity import BaseActivity, SuspiciousEvent
from pipeline.spatial_normalizer import NormalizedPose


class VelocityDetector(BaseActivity):
    """Detecta velocidade de movimento anormal baseada em 'torsos por segundo',
    tornando-a independente de resolução (pixels) e framerate (FPS)."""

    def __init__(self, velocidade_maxima: float = 5.0, cooldown_segundos: float = 2.0):
        super().__init__("velocidade", threshold=0.6)
        self.velocidade_maxima = velocidade_maxima
        self.cooldown_segundos = cooldown_segundos

        # Histórico indexado por track_id
        self.ultima_posicao = {}            # track_id -> pelvis (x, y)
        self.ultimo_timestamp = {}          # track_id -> timestamp (float)
        self.ultimo_alerta = {}             # track_id -> timestamp do último alerta

    def limpa_tracks_inativas(self, ids_presentes: set):
        """Limpa o histórico de tracks inativas para evitar vazamento de memória."""
        for track_id in list(self.ultima_posicao.keys()):
            if track_id not in ids_presentes:
                self.ultima_posicao.pop(track_id, None)
                self.ultimo_timestamp.pop(track_id, None)
                self.ultimo_alerta.pop(track_id, None)

    def detecta(self,
                norm_pose: NormalizedPose,
                frame_id: int,
                timestamp: float,
                track_id: Optional[int] = None) -> Optional[SuspiciousEvent]:
        tid = 0 if track_id is None else track_id

        if not norm_pose or not norm_pose.is_valid:
            return None

        pelvis_atual = norm_pose.pelvis
        torso_length = norm_pose.torso_length

        if torso_length <= 0:
            return None

        # Se é o primeiro frame da track, inicializa os dados
        if tid not in self.ultima_posicao:
            self.ultima_posicao[tid] = pelvis_atual
            self.ultimo_timestamp[tid] = timestamp
            self.ultimo_alerta[tid] = 0.0
            return None

        # Calcula o tempo e distância percorrida
        tempo_decorrido = timestamp - self.ultimo_timestamp[tid]
        
        # Evita divisão por zero ou updates no mesmo timestamp
        if tempo_decorrido <= 0:
            return None

        distancia_pixels = np.linalg.norm(pelvis_atual - self.ultima_posicao[tid])
        
        # Velocidade em Torsos por Segundo (Independente de Pixels e FPS)
        # distancia / torso_length = distância em 'torsos'
        distancia_torsos = distancia_pixels / torso_length
        velocidade_instantanea = distancia_torsos / tempo_decorrido

        # Atualiza histórico para o próximo frame
        self.ultima_posicao[tid] = pelvis_atual
        self.ultimo_timestamp[tid] = timestamp

        # Verifica cooldown
        tempo_desde_alerta = timestamp - self.ultimo_alerta.get(tid, 0.0)

        if velocidade_instantanea > self.velocidade_maxima and tempo_desde_alerta >= self.cooldown_segundos:
            self.ultimo_alerta[tid] = timestamp

            confianca = min(velocidade_instantanea / (self.velocidade_maxima * 1.5), 1.0)
            
            evento = SuspiciousEvent(
                tipo="velocidade",
                timestamp=timestamp,
                confianca=confianca,
                frame_id=frame_id,
                pessoa_id=track_id,
                descricao=f"Movimento rápido detectado: {velocidade_instantanea:.1f} torsos/s",
                dados_adicionais={
                    'velocidade_torsos_seg': float(velocidade_instantanea),
                    'torso_length_px': float(torso_length),
                    'threshold_torsos_seg': float(self.velocidade_maxima)
                }
            )
            self.registra_evento(evento)
            return evento

        return None

