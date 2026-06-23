from pydantic import BaseModel

class ZonaSincronizada(BaseModel):
    """Schema para sincronização de eventos de zona do Edge."""
    track_id: int
    zone_id: int
    zone_name: str
    hand: str
    deceleration_ratio: float
    arm_flex_ratio: float
    arm_length: float
    timestamp: str
