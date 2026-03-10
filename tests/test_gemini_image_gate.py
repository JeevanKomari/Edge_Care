import io
import json

import pytest
import requests
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.services import gemini_image_gate
from app.services.gemini_image_gate import reset_gate_metrics, run_image_gate


def _make_image_bytes(color=(220, 180, 180)) -> bytes:
    img = Image.new("RGB", (256, 256), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _fake_response(label: str, confidence: float = 0.8, reason: str = ""):
    text = json.dumps({"label": label, "confidence": confidence, "reason": reason})
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": text}
                    ]
                }
            }
        ]
    }


def test_gate_accepts_rash(monkeypatch):
    reset_gate_metrics()
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")

    class Resp:
        status_code = 200
        def json(self):
            return _fake_response("rash_like_skin", 0.9, "looks like rash")
        def raise_for_status(self):
            return None

    monkeypatch.setattr(requests, "post", lambda *a, **k: Resp())
    res = run_image_gate(b"abc", filename="f.png", content_type="image/png", request_id="req1")
    assert res["status"] == "accepted"
    assert res["accepted"] is True
    assert res["top_label"] == "rash_like_skin"


def test_gate_rejects_non_skin(monkeypatch):
    reset_gate_metrics()
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")

    class Resp:
        status_code = 200
        def json(self):
            return _fake_response("non_skin_or_irrelevant_image", 0.85, "object")
        def raise_for_status(self):
            return None

    monkeypatch.setattr(requests, "post", lambda *a, **k: Resp())
    res = run_image_gate(b"abc")
    assert res["status"] == "rejected"
    assert res["accepted"] is False


def test_gate_rejects_low_quality(monkeypatch):
    reset_gate_metrics()
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")

    class Resp:
        status_code = 200
        def json(self):
            return _fake_response("unclear_or_low_quality", 0.6, "blurry")
        def raise_for_status(self):
            return None

    monkeypatch.setattr(requests, "post", lambda *a, **k: Resp())
    res = run_image_gate(b"abc")
    assert res["status"] == "rejected"
    assert res["accepted"] is False


def test_fail_open_on_timeout(monkeypatch):
    reset_gate_metrics()
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")

    def raise_timeout(*a, **k):
        raise requests.Timeout()

    monkeypatch.setattr(requests, "post", raise_timeout)
    res = run_image_gate(b"abc")
    assert res["status"] == "fail_open"
    assert res["accepted"] is True
    assert "GATE_TIMEOUT" in res.get("reason_codes", [])
    assert res["latency_ms"] >= 0


def test_retry_on_503_then_success(monkeypatch):
    reset_gate_metrics()
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    calls = {"n": 0}

    class Resp503:
        status_code = 503
        text = "unavailable"
        def json(self): return {"error": "svc"}
        def raise_for_status(self): raise requests.HTTPError(response=self)
    class Resp200:
        status_code = 200
        def json(self): return _fake_response("rash_like_skin", 0.9, "ok")
        def raise_for_status(self): return None
        text = json.dumps(_fake_response("rash_like_skin", 0.9, "ok"))

    def post(*a, **k):
        calls["n"] += 1
        return Resp503() if calls["n"] == 1 else Resp200()

    monkeypatch.setattr(requests, "post", post)
    res = run_image_gate(b"abc")
    assert res["status"] == "accepted"
    assert res["attempted_models"] == [res["model_id"]]
    snap = gemini_image_gate.metrics.snapshot()
    assert snap["counters"]["gate_retry_total"] >= 1


def test_parse_failure(monkeypatch):
    reset_gate_metrics()
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    class Resp:
        status_code = 200
        text = "badjson"
        def json(self): return {"candidates": [{"content": {"parts": [{"text": "not json"}]}}]}
        def raise_for_status(self): return None

    monkeypatch.setattr(requests, "post", lambda *a, **k: Resp())
    res = run_image_gate(b"abc")
    assert res["status"] in ("fail_open", "error")
    assert "GATE_JSON_PARSE_ERROR" in res.get("reason_codes", [])
    assert res["latency_ms"] >= 0


def test_gate_disabled(monkeypatch):
    reset_gate_metrics()
    monkeypatch.setenv("GEMINI_GATE_ENABLED", "false")
    res = run_image_gate(b"abc")
    assert res["status"] == "disabled"
    assert res["accepted"] is True


def test_endpoint_runs_pipeline_when_gate_accepts(monkeypatch):
    client = TestClient(app)
    img_bytes = _make_image_bytes()
    stub_gate = {
        "enabled": True,
        "status": "accepted",
        "accepted": True,
        "model_id": "gemini-stub",
        "latency_ms": 10,
        "top_label": "rash_like_skin",
        "top_score": 0.9,
        "scores": [],
        "reasons": [],
        "thresholds": {},
    }
    monkeypatch.setattr("app.ml.image_model.run_image_gate", lambda *a, **k: stub_gate)
    monkeypatch.setattr(
        "app.ml.image_model._predict_severity",
        lambda data: (
            {"predicted_class": "mild", "confidence": 0.9, "all_probabilities": {"mild": 0.9, "moderate": 0.05, "severe": 0.05}},
            None,
        ),
    )
    monkeypatch.setattr(
        "app.ml.image_model._predict_disease",
        lambda data: {"status": "classified", "predicted_class": "eczema", "confidence": 0.8, "top_predictions": [], "all_probabilities": {}},
    )
    monkeypatch.setattr("app.ml.image_model.is_image_blurry", lambda img: False)

    resp = client.post("/ml/analyze-image", files={"file": ("img.png", img_bytes, "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["image_gate"]["status"] == "accepted"
    assert body["ml_analysis"]["predicted_class"] == "mild"


def test_endpoint_skips_ml_when_gate_rejects(monkeypatch):
    client = TestClient(app)
    img_bytes = _make_image_bytes()
    stub_gate = {
        "enabled": True,
        "status": "rejected",
        "accepted": False,
        "model_id": "gemini-stub",
        "latency_ms": 10,
        "top_label": "non_skin_or_irrelevant_image",
        "top_score": 0.9,
        "scores": [],
        "reasons": ["not skin"],
        "thresholds": {},
    }
    monkeypatch.setattr("app.ml.image_model.run_image_gate", lambda *a, **k: stub_gate)
    monkeypatch.setattr("app.ml.image_model.is_image_blurry", lambda img: False)

    resp = client.post("/ml/analyze-image", files={"file": ("img.png", img_bytes, "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is False
    assert body["image_gate"]["status"] == "rejected"
    assert body["ml_analysis"] is None
