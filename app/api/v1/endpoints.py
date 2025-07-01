from fastapi import APIRouter, HTTPException, Request
from app.api.v1.schemas import IntentRequest, IntentResponse, Prediction

router = APIRouter()

@router.post("", response_model=IntentResponse)
async def classify_intent(request: Request, payload: IntentRequest):
    text = payload.text.strip()
    if not text:
        raise HTTPException(
            status_code=400,
            detail={"label": "TEXT_EMPTY", "message": "\"text\" is empty."}
        )

    model = request.app.state.model
    if not model or not model.is_ready():
        raise HTTPException(status_code=423, detail="Not ready")

    try:
        results = model.predict(text)
        preds = [Prediction(label=l, confidence=c) for l, c in results]
        return IntentResponse(intents=preds)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"label": "INTERNAL_ERROR", "message": str(e)}
        )
