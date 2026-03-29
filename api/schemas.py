"""
api/schemas.py
--------------
Pydantic models for request validation and response serialization.
FastAPI uses these automatically for /docs and input validation.
"""

from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class ScoreRequest(BaseModel):
    user_id: str

    class Config:
        json_schema_extra = {
            "example": {"user_id": "U_042"}
        }


class FeatureContribution(BaseModel):
    feature:     str
    description: str
    value:       float
    importance:  float


class ScoreResponse(BaseModel):
    user_id:       str
    risk_score:    float           # ensemble score 0-1
    risk_label:    str             # low / medium / high
    if_score:      Optional[float] # isolation forest
    lstm_score:    Optional[float] # lstm reconstruction error
    top_features:  list[FeatureContribution]
    alert_sent:    bool
    scored_at:     datetime


class AlertResponse(BaseModel):
    id:          int
    user_id:     str
    alert_type:  Optional[str]
    risk_score:  float
    message:     str
    created_at:  datetime
    acknowledged: bool


class HealthResponse(BaseModel):
    status:      str
    db:          str
    rabbitmq:    str
    models_loaded: bool