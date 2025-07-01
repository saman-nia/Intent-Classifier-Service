import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_intent_empty(monkeypatch):
    class Dummy:
        def is_ready(self): return True
        def predict(self, text): return []
    monkeypatch.setattr(app.state, "model", Dummy())

    resp = client.post("/intent", json={"text": ""})
    assert resp.status_code == 400
    assert resp.json() == {"label": "TEXT_EMPTY", "message": "\"text\" is empty."}

def test_intent_success(monkeypatch):
    class Dummy:
        def is_ready(self): return True
        def predict(self, text):
            return [("flight", 0.73), ("aircraft", 0.12), ("capacity", 0.03)]
    monkeypatch.setattr(app.state, "model", Dummy())

    resp = client.post("/intent", json={"text": "fly me"})
    assert resp.status_code == 200
    assert resp.json() == {
        "intents": [
            {"label": "flight", "confidence": 0.73},
            {"label": "aircraft", "confidence": 0.12},
            {"label": "capacity", "confidence": 0.03},
        ]
    }
