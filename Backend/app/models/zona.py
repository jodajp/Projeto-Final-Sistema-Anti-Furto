from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func
from app.database import Base

class ZonaModel(Base):
    """Modelo relacional para eventos de grab em zonas de prateleira."""
    __tablename__ = "zonas"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, index=True)
    track_id = Column(Integer, nullable=False)
    zone_id = Column(Integer, nullable=False)
    zone_name = Column(String(100), nullable=False)
    hand = Column(String(20), nullable=False)
    deceleration_ratio = Column(Float, nullable=False)
    arm_flex_ratio = Column(Float, nullable=False)
    arm_length = Column(Float, nullable=True)
