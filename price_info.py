from pydantic import BaseModel

class NAS100(BaseModel):
    open: float
    volume: float
    low: float
    high: float

class US30(BaseModel):
    open: float
    volume: float
    low: float
    high: float

class GER30(BaseModel):
    open: float
    volume: float
    low: float
    high: float