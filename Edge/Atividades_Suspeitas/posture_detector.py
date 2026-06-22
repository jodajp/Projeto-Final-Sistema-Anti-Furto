"""
Detector de Agachamento
Detecta quando a pessoa está agachada (típico de roubo em prateleiras baixas) usando ângulos de articulação.
"""

from typing import Optional
import numpy as np

from .base_activity import BaseActivity, SuspiciousEvent
from Detecao.skeleton import (
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE
)
from pipeline.spatial_normalizer import NormalizedPose


def calcula_angulo(p1, p2, p3):
    """Calcula o ângulo interior formado pelos pontos p1-p2-p3 (com vértice p2) em graus."""
    v1 = p1 - p2
    v2 = p3 - p2
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-5 or norm2 < 1e-5:
        return 180.0  # Retorna esticado por defeito se falhar
    cos_theta = np.dot(v1, v2) / (norm1 * norm2)
    return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))


class PostureDetector(BaseActivity):
    """Detecta postura (agachamento) usando ângulos físicos das pernas e quadris."""

    def __init__(self, agachamento_threshold: float = 120.0, tempo_minimo: int = 5, cooldown_frames: int = 60):
        # Se o threshold passado for no formato do antigo ratio (geralmente < 2.0, ex: 0.6),
        # usamos o padrão de 120.0 graus para a lógica de ângulos físicos.
        threshold_graus = 120.0 if agachamento_threshold < 2.0 else agachamento_threshold
        super().__init__("agachamento", threshold=threshold_graus)
        self.tempo_minimo = tempo_minimo
        self.cooldown_frames = cooldown_frames

        # Estado por track_id
        self.frames_agachado = {}           # track_id -> frames consecutivos agachado
        self.frames_since_last_alerts = {}  # track_id -> int (cooldown tracker)

    def limpa_tracks_inativas(self, ids_presentes: set):
        """Limpa o histórico de tracks inativas para evitar vazamento de memória."""
        for track_id in list(self.frames_agachado.keys()):
            if track_id not in ids_presentes:
                self.frames_agachado.pop(track_id, None)
                self.frames_since_last_alerts.pop(track_id, None)

    def detecta(self,
                norm_pose: NormalizedPose,
                frame_id: int,
                timestamp: float,
                track_id: Optional[int] = None) -> Optional[SuspiciousEvent]:
        tid = 0 if track_id is None else track_id

        if tid not in self.frames_agachado:
            self.frames_agachado[tid] = 0
            self.frames_since_last_alerts[tid] = self.cooldown_frames

        self.frames_since_last_alerts[tid] += 1

        if not norm_pose or not norm_pose.is_valid:
            self.frames_agachado[tid] = 0
            return None

        kp = norm_pose.keypoints
        sc = norm_pose.scores

        # 1. TENTA DETETAR USANDO O ÂNGULO DO JOELHO (Melhor caso)
        angulos_joelho = []
        if sc[LEFT_HIP] > 0.3 and sc[LEFT_KNEE] > 0.3 and sc[LEFT_ANKLE] > 0.3:
            angulos_joelho.append(calcula_angulo(kp[LEFT_HIP], kp[LEFT_KNEE], kp[LEFT_ANKLE]))
        if sc[RIGHT_HIP] > 0.3 and sc[RIGHT_KNEE] > 0.3 and sc[RIGHT_ANKLE] > 0.3:
            angulos_joelho.append(calcula_angulo(kp[RIGHT_HIP], kp[RIGHT_KNEE], kp[RIGHT_ANKLE]))

        is_agachado = False
        angulo_detetado = 180.0
        tipo_angulo = "joelho"

        if angulos_joelho:
            angulo_detetado = min(angulos_joelho)
            if angulo_detetado < self.threshold:
                is_agachado = True
        else:
            # 2. FALLBACK: CASO DE OCLUSÃO DOS TORNOZELOS (Usa ângulo dos quadris / tronco-coxa)
            angulos_quadril = []
            if sc[LEFT_SHOULDER] > 0.3 and sc[LEFT_HIP] > 0.3 and sc[LEFT_KNEE] > 0.3:
                angulos_quadril.append(calcula_angulo(kp[LEFT_SHOULDER], kp[LEFT_HIP], kp[LEFT_KNEE]))
            if sc[RIGHT_SHOULDER] > 0.3 and sc[RIGHT_HIP] > 0.3 and sc[RIGHT_KNEE] > 0.3:
                angulos_quadril.append(calcula_angulo(kp[RIGHT_SHOULDER], kp[RIGHT_HIP], kp[RIGHT_KNEE]))

            if angulos_quadril:
                angulo_detetado = min(angulos_quadril)
                if angulo_detetado < self.threshold:
                    is_agachado = True
                    tipo_angulo = "quadril (torso/coxa)"

        # 3. AVALIA A PERSISTÊNCIA DO AGACHAMENTO
        if is_agachado:
            self.frames_agachado[tid] += 1

            if (self.frames_agachado[tid] >= self.tempo_minimo and
                    self.frames_since_last_alerts[tid] >= self.cooldown_frames):
                self.frames_since_last_alerts[tid] = 0
                confianca = float(np.clip((self.threshold - angulo_detetado) / 60.0 + 0.5, 0.5, 1.0))

                evento = SuspiciousEvent(
                    tipo="agachamento",
                    timestamp=timestamp,
                    confianca=confianca,
                    frame_id=frame_id,
                    pessoa_id=track_id,
                    descricao=f"Agachamento detetado: Angulo do {tipo_angulo} a {angulo_detetado:.1f}",
                    dados_adicionais={
                        'angulo': float(angulo_detetado),
                        'tipo_angulo': tipo_angulo,
                        'threshold': float(self.threshold),
                        'frames_agachado': int(self.frames_agachado[tid])
                    }
                )
                self.registra_evento(evento)
                return evento
        else:
            self.frames_agachado[tid] = 0

        return None
