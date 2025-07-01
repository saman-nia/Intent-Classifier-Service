from pydantic import BaseModel
from typing import List

class IntentRequest(BaseModel):
    text: str

class Prediction(BaseModel):
    label: str
    confidence: float

class IntentResponse(BaseModel):
    intents: List[Prediction]
