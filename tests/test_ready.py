import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_ready_not_loaded(monkeypatch):
    monkeypatch.setattr(app.state, "model", None)
    resp = client.get("/ready")
    assert resp.status_code == 423
    assert resp.json() == {"label": "INTERNAL_ERROR", "message": "Not ready"}

def test_ready_loaded(monkeypatch):
    class Dummy:
        def is_ready(self): return True
    monkeypatch.setattr(app.state, "model", Dummy())
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.text.strip('"') == "OK"
