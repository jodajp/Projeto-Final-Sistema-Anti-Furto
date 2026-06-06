from pydantic import BaseModel

class AlertaSincronizado(BaseModel):
    track_id: int
    tipo_alerta: str
    confianca: float
    timestamp: str