from typing import Optional, Dict
import numpy as np

from .base_activity import BaseActivity, SuspiciousEvent
from Detecao.skeleton import LEFT_WRIST, RIGHT_WRIST
from pipeline.spatial_normalizer import NormalizedPose


class VelocityDetector(BaseActivity):
    """Detecta velocidade de movimento anormal (corridas ou gestos rápidos como roubos)
    baseando-se em 'torsos por segundo', tornando o cálculo independente de resolução e FPS."""

    def __init__(self, velocidade_maxima: float = 5.0, velocidade_maxima_mao: float = 8.0, cooldown_segundos: float = 2.0):
        super().__init__("velocidade", threshold=0.6)
        self.velocidade_maxima = velocidade_maxima
        self.velocidade_maxima_mao = velocidade_maxima_mao
        self.cooldown_segundos = cooldown_segundos

        # Histórico indexado por track_id -> dicionário de juntas {'pelvis': pos, 'l_wrist': pos, 'r_wrist': pos}
        self.ultima_posicao = {}            # track_id -> dict
        self.ultimo_timestamp = {}          # track_id -> timestamp (float)
        self.ultimo_alerta = {}             # track_id -> timestamp do último alerta

    def limpa_tracks_inativas(self, ids_presentes: set):
        """Limpa o histórico de tracks inativas para evitar vazamento de memória."""
        for track_id in list(self.ultima_posicao.keys()):
            if track_id not in ids_presentes:
                self.ultima_posicao.pop(track_id, None)
                self.ultimo_timestamp.pop(track_id, None)
                self.ultimo_alerta.pop(track_id, None)

    def _get_raw_positions(self, norm_pose: NormalizedPose) -> Dict[str, np.ndarray]:
        positions = {'pelvis': norm_pose.pelvis}
        
        # Reconstrói a posição absoluta em pixels para os pulsos a partir da pose normalizada
        kp = norm_pose.keypoints
        sc = norm_pose.scores
        torso_len = norm_pose.torso_length
        
        if sc[LEFT_WRIST] >= 0.35:
            positions['l_wrist'] = kp[LEFT_WRIST] * torso_len + norm_pose.pelvis
        if sc[RIGHT_WRIST] >= 0.35:
            positions['r_wrist'] = kp[RIGHT_WRIST] * torso_len + norm_pose.pelvis
            
        return positions

    def detecta(self,
                norm_pose: NormalizedPose,
                frame_id: int,
                timestamp: float,
                track_id: Optional[int] = None) -> Optional[SuspiciousEvent]:
        tid = 0 if track_id is None else track_id

        if not norm_pose or not norm_pose.is_valid:
            return None

        torso_length = norm_pose.torso_length
        if torso_length <= 0:
            return None

        posicoes_atuais = self._get_raw_positions(norm_pose)

        # Se é o primeiro frame da track, inicializa os dados
        if tid not in self.ultima_posicao:
            self.ultima_posicao[tid] = posicoes_atuais
            self.ultimo_timestamp[tid] = timestamp
            self.ultimo_alerta[tid] = 0.0
            return None

        # Calcula o tempo decorrido
        tempo_decorrido = timestamp - self.ultimo_timestamp[tid]
        
        # Evita divisão por zero ou updates no mesmo timestamp
        if tempo_decorrido <= 0:
            return None

        posicoes_antigas = self.ultima_posicao[tid]
        velocidades = {}

        # Calcula velocidades para cada junta disponível em ambos os frames
        for joint in ['pelvis', 'l_wrist', 'r_wrist']:
            if joint in posicoes_atuais and joint in posicoes_antigas:
                dist_px = np.linalg.norm(posicoes_atuais[joint] - posicoes_antigas[joint])
                dist_torsos = dist_px / torso_length
                velocidades[joint] = dist_torsos / tempo_decorrido

        # Atualiza histórico para o próximo frame
        self.ultima_posicao[tid] = posicoes_atuais
        self.ultimo_timestamp[tid] = timestamp

        # Verifica cooldown
        tempo_desde_alerta = timestamp - self.ultimo_alerta.get(tid, 0.0)
        if tempo_desde_alerta < self.cooldown_segundos:
            return None

        # Verifica se algum ultrapassou seu respectivo threshold
        alerta_trigger = False
        descricao_alerta = ""
        max_ratio = 0.0

        if 'pelvis' in velocidades:
            vel = velocidades['pelvis']
            ratio = vel / self.velocidade_maxima
            if ratio > 1.0 and ratio > max_ratio:
                max_ratio = ratio
                alerta_trigger = True
                descricao_alerta = f"Movimento corporal rápido: {vel:.1f} torsos/s"

        for joint_name, joint_key in [("Mão esquerda", "l_wrist"), ("Mão direita", "r_wrist")]:
            if joint_key in velocidades:
                vel = velocidades[joint_key]
                ratio = vel / self.velocidade_maxima_mao
                if ratio > 1.0 and ratio > max_ratio:
                    max_ratio = ratio
                    alerta_trigger = True
                    descricao_alerta = f"{joint_name} com movimento brusco: {vel:.1f} torsos/s"

        if alerta_trigger:
            self.ultimo_alerta[tid] = timestamp
            confianca = min(max_ratio * 0.5, 1.0)
            
            evento = SuspiciousEvent(
                tipo="velocidade",
                timestamp=timestamp,
                confianca=confianca,
                frame_id=frame_id,
                pessoa_id=track_id,
                descricao=descricao_alerta,
                dados_adicionais={
                    'velocidades': {k: float(v) for k, v in velocidades.items()},
                    'torso_length_px': float(torso_length),
                    'threshold_pelvis': float(self.velocidade_maxima),
                    'threshold_mao': float(self.velocidade_maxima_mao)
                }
            )
            self.registra_evento(evento)
            return evento

        return None

