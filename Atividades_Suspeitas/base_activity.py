"""
Classe base abstrata para detectores de atividades suspeitas
Permite adicionar novos tipos de comportamento suspeito facilmente
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, List
import numpy as np

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
        self.historico = []
        
    @abstractmethod
    def detecta(self, 
                keypoints: List[tuple], 
                scores: List[float],
                frame_id: int,
                timestamp: float) -> Optional[SuspiciousEvent]:
        """
        Detecta atividade suspeita.
        
        Args:
            keypoints: Lista de (x, y) para cada keypoint
            scores: Confiança de cada keypoint
            frame_id: ID do frame
            timestamp: Timestamp do frame
            
        Returns:
            SuspiciousEvent se detectou algo, None caso contrário
        """
        pass
    
    def registra_evento(self, evento: SuspiciousEvent):
        """Registra evento no histórico."""
        self.historico.append(evento)
    
    def get_historico(self) -> List[SuspiciousEvent]:
        """Retorna histórico de eventos."""
        return self.historico
    
    def limpa_historico(self):
        """Limpa o histórico."""
        self.historico = []
