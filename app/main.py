import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.api.v1.endpoints import router as intent_router
from app.core.config import settings
from app.services.intent_classifier import IntentClassifier
from app.services.bert_classifier   import BertIntentClassifier
from app.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Intent Classification Service",
    version="0.1.0",
    description="FastAPI service for intent classification."
)

@app.on_event("startup")
async def load_model():
    logger.info("Loading model from %s", settings.model_path)
    impl = os.getenv("MODEL_IMPL", "fasttext").lower()
    cls  = {"fasttext": IntentClassifier, "bert": BertIntentClassifier}[impl]()
    try:
        cls.load(settings.model_path)
        app.state.model = cls
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.exception("Failed to load model: %s", e)
        app.state.model = None

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if isinstance(exc.detail, dict) and "label" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"label": "INTERNAL_ERROR", "message": str(exc.detail)}
    )

@app.get("/ready")
async def ready():
    model = getattr(app.state, "model", None)
    if model and model.is_ready():
        return "OK"
    raise HTTPException(status_code=423, detail="Not ready")

app.include_router(intent_router, prefix="/intent", tags=["intent"])
