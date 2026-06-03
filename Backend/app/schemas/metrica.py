from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime
from app.database import Base
from datetime import datetime

# O formato exato do JSON que o teu Edge gera
class MetricaNodeBase(BaseModel):
    node_id: str
    timestamp: Optional[float] = None
    fps: float
    frame_count: int
    detection_count: int
    inference_calls: int
    average_inference_ms: float
    success_rate: float
    uptime_seconds: float
    pessoas_detetadas: int = 0

class MetricaNodeCreate(MetricaNodeBase):
    pass

class MetricaNodeResponse(MetricaNodeBase):
    id: int
    data_recebida: datetime

    class Config:
        from_attributes = True

class ClusterMetricsSummary(BaseModel):
    num_nodes: int
    media_fps: float
    total_frames: int
    total_detections: int
    total_inference_calls: int
    tempo_medio_inferencia_ms: float
    taxa_sucesso_media_pct: float
    uptime_maximo_segundos: int

class MetricasClusterResponse(BaseModel):
    cluster_metrics: Optional[ClusterMetricsSummary]
    nodes: List[MetricaNodeBase]
    timestamp: str