from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func
from app.database import Base

class AlertaModel(Base):
    """Modelo relacional para mapeamento de eventos de furto no PostgreSQL."""
    __tablename__ = "alertas"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, index=True)
    track_id = Column(Integer, nullable=False)
    tipo_alerta = Column(String(50), nullable=False)
    confianca = Column(Float, nullable=False)