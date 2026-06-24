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
    """Detecta postura (agachamento) usando a extensão vertical do corpo (compressão da pose)."""

    def __init__(self, agachamento_threshold: float = 1.3, tempo_minimo: int = 5, cooldown_frames: int = 60):
        super().__init__("agachamento", threshold=agachamento_threshold)
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

        # Em vez de ângulos articulares que são complexos e propensos a ruído (dobrar uma perna),
        # medimos a compressão vertical do corpo.
        # Na pose normalizada, o pélvis é (0,0) e a escala é o tamanho do torso.
        # A coordenada Y aumenta para baixo. Logo, a extensão da perna dita se a pessoa está em pé.
        
        y_ankles = [kp[idx][1] for idx in (LEFT_ANKLE, RIGHT_ANKLE) if sc[idx] > 0.3]
        y_knees  = [kp[idx][1] for idx in (LEFT_KNEE, RIGHT_KNEE) if sc[idx] > 0.3]

        is_agachado = False
        extensao_medida = 0.0
        tipo_extensao = "nenhuma"

        # Thresholds calibrados para o tamanho do torso.
        threshold_tornozelo = self.threshold
        threshold_joelho = self.threshold - 0.7  # O joelho é tipicamente 0.7 torsos mais acima que o tornozelo

        if y_ankles:
            # A extensão máxima da perna é garantida pelo pé que estiver MAIS ESTICADO (maior Y).
            # Uma pessoa em pé tem max(y_ankles) entre ~1.8 e ~2.0.
            # Se a pessoa levantar UMA perna, a outra garante que max(y_ankles) continua ~1.8 (evita falsos positivos).
            extensao_medida = max(y_ankles)
            if extensao_medida < threshold_tornozelo:
                is_agachado = True
            tipo_extensao = "tornozelo"
        elif y_knees:
            # Fallback seguro caso os tornozelos estejam ocluídos por prateleiras.
            # Uma pessoa em pé tem max(y_knees) perto de ~1.0.
            extensao_medida = max(y_knees)
            if extensao_medida < threshold_joelho:
                is_agachado = True
            tipo_extensao = "joelho"

        # AVALIAR A PERSISTÊNCIA DO AGACHAMENTO
        if is_agachado:
            self.frames_agachado[tid] += 1

            if (self.frames_agachado[tid] >= self.tempo_minimo and
                    self.frames_since_last_alerts[tid] >= self.cooldown_frames):
                self.frames_since_last_alerts[tid] = 0
                
                # Confiança inversamente proporcional à extensão (mais encolhido = maior confiança)
                margem = (threshold_tornozelo if tipo_extensao == "tornozelo" else threshold_joelho)
                confianca = float(np.clip(1.0 - (extensao_medida / margem) * 0.5, 0.5, 1.0))

                evento = SuspiciousEvent(
                    tipo="agachamento",
                    timestamp=timestamp,
                    confianca=confianca,
                    frame_id=frame_id,
                    pessoa_id=track_id,
                    descricao=f"Agachamento detetado: Extensao vertical ({tipo_extensao}) encolhida para {extensao_medida:.2f}x o torso",
                    dados_adicionais={
                        'extensao_medida': float(extensao_medida),
                        'tipo_extensao': tipo_extensao,
                        'threshold_usado': float(margem),
                        'frames_agachado': int(self.frames_agachado[tid])
                    }
                )
                self.registra_evento(evento)
                return evento
        else:
            self.frames_agachado[tid] = 0

        return None
