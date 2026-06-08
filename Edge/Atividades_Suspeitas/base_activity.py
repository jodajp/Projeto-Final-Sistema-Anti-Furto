"""
Classe base abstrata para detectores de atividades suspeitas.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from pipeline.spatial_normalizer import NormalizedPose


@dataclass
class SuspiciousEvent:
    """Evento suspeito detectado."""
    tipo: str                      # 'velocidade', 'agachamento', etc
    timestamp: float
    confianca: float               # 0.0-1.0
    frame_id: int
    pessoa_id: Optional[int] = None
    descricao: str = ""
    dados_adicionais: Dict = None


class BaseActivity(ABC):
    """Classe abstrata para todos os detectores de atividade."""

    def __init__(self, nome: str, threshold: float = 0.5):
        self.nome = nome
        self.threshold = threshold
        self.historico: List[SuspiciousEvent] = []

    @abstractmethod
    def detecta(self,
                norm_pose: NormalizedPose,
                frame_id: int,
                timestamp: float,
                track_id: Optional[int] = None) -> Optional[SuspiciousEvent]:
        """
        Detecta atividade suspeita usando a pose normalizada.

        Args:
            norm_pose: Objeto contendo os keypoints normalizados e metadados.
            frame_id: ID do frame.
            timestamp: Timestamp do frame.
            track_id: ID do rastreamento (ByteTrack).

        Returns:
            SuspiciousEvent se detectou algo, None caso contrário.
        """
        pass

    def limpa_tracks_inativas(self, ids_presentes: set):
        """Limpa o estado armazenado para tracks que não estão mais presentes."""
        pass

    def registra_evento(self, evento: SuspiciousEvent):
        """Regista evento no histórico local e envia para o backend via orchestrator."""
        self.historico.append(evento)

    def get_historico(self) -> List[SuspiciousEvent]:
        """Retorna histórico de eventos desta sessão."""
        return self.historico

    def limpa_historico(self):
        """Limpa o histórico."""
        self.historico = []
