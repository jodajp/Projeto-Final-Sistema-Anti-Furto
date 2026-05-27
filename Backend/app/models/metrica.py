from sqlalchemy import Column, Integer, String, Float, DateTime
from app.database import Base
from datetime import datetime

class MetricaNodeModel(Base):
    __tablename__ = "metricas_performance"

    id = Column(Integer, primary_key=True, index=True)
    node_id = Column(String, index=True)
    timestamp = Column(Float)
    fps = Column(Float)
    frame_count = Column(Integer)
    detection_count = Column(Integer)
    inference_calls = Column(Integer)
    average_inference_ms = Column(Float)
    success_rate = Column(Float)
    uptime_seconds = Column(Float)
    data_recebida = Column(DateTime, default=datetime.utcnow)