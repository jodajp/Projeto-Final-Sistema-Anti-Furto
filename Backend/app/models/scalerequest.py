from pydantic import BaseModel

class ScaleRequest(BaseModel):
    replicas: int