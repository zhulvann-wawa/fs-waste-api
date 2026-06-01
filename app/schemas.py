from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])
    model_loaded: bool


class ClassScore(BaseModel):
    label: str
    confidence: float


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    predictions: list[ClassScore]


class ModelInfoResponse(BaseModel):
    model_type: str
    image_size: int
    classes: list[str]


class ClassesResponse(BaseModel):
    classes: list[str]
