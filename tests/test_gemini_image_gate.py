import io
import asyncio
import json as jsonlib
import time

import httpx
import pytest
import requests
from PIL import Image

from app.main import app
from app.services import gemini_image_gate
from app.services.gemini_disease_identifier import run_disease_identifier
from app.services.gemini_image_gate import reset_gate_metrics, run_image_gate


def _make_image_bytes(color=(220, 180, 180)) -> bytes:
    img = Image.new("RGB", (256, 256), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _fake_response(label: str, confidence: float = 0.8, reason: str = ""):
    text = jsonlib.dumps({"label": label, "confidence": confidence, "reason": reason})
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


def _post_analyze_image(img_bytes: bytes):
    async def _send():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post("/ml/analyze-image", files={"file": ("img.png", img_bytes, "image/png")})

    return asyncio.run(_send())


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
        text = jsonlib.dumps(_fake_response("rash_like_skin", 0.9, "ok"))

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


def test_slow_gate_flag(monkeypatch):
    reset_gate_metrics()
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")

    class Resp:
        status_code = 200
        def json(self): return _fake_response("rash_like_skin", 0.9, "ok")
        def raise_for_status(self): return None
        text = jsonlib.dumps(_fake_response("rash_like_skin", 0.9, "ok"))

    monkeypatch.setattr(requests, "post", lambda *a, **k: Resp())
    ticks = [0.0]
    def fake_perf_counter():
        ticks[0] += 9.1  # simulate 9.1 seconds total
        return ticks[0]
    monkeypatch.setattr(time, "perf_counter", fake_perf_counter)
    res = run_image_gate(b"abc")
    assert res.get("slow_gate") is True


def test_normal_gate_not_slow(monkeypatch):
    reset_gate_metrics()
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")

    class Resp:
        status_code = 200
        def json(self): return _fake_response("rash_like_skin", 0.9, "ok")
        def raise_for_status(self): return None
        text = jsonlib.dumps(_fake_response("rash_like_skin", 0.9, "ok"))

    monkeypatch.setattr(requests, "post", lambda *a, **k: Resp())
    ticks = [0.0]
    def fake_perf_counter():
        ticks[0] += 1.5
        return ticks[0]
    monkeypatch.setattr(time, "perf_counter", fake_perf_counter)
    res = run_image_gate(b"abc")
    assert res.get("slow_gate") in (False, None)


def test_gate_disabled(monkeypatch):
    reset_gate_metrics()
    monkeypatch.setenv("GEMINI_GATE_ENABLED", "false")
    res = run_image_gate(b"abc")
    assert res["status"] == "disabled"
    assert res["accepted"] is True


def test_disease_identifier_uses_gemini_and_normalizes(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")

    class Resp:
        status_code = 200
        def json(self):
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": jsonlib.dumps(
                                        {
                                            "condition": "ringworm",
                                            "confidence": 0.83,
                                            "differentials": ["eczema", "psoriasis"],
                                            "reason": "Annular scaly border is visible.",
                                            "visible_skin_concern": True,
                                        }
                                    )
                                }
                            ]
                        }
                    }
                ]
            }
        def raise_for_status(self):
            return None

    monkeypatch.setattr(requests, "post", lambda *a, **k: Resp())
    result = run_disease_identifier(b"abc", request_id="req-disease")
    assert result["status"] == "classified"
    assert result["source"] == "gemini"
    assert result["display_name"] == "Fungal Infection"
    assert result["canonical_label"] == "Tinea Ringworm Candidiasis and other Fungal Infections"


def test_disease_identifier_marks_weak_visible_match_provisional(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")

    class Resp:
        status_code = 200
        def json(self):
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": jsonlib.dumps(
                                        {
                                            "condition": "dermatitis",
                                            "confidence": 0.38,
                                            "differentials": ["eczema"],
                                            "reason": "Visible inflamed skin pattern, but exact type is weak.",
                                            "visible_skin_concern": True,
                                        }
                                    )
                                }
                            ]
                        }
                    }
                ]
            }
        def raise_for_status(self):
            return None

    monkeypatch.setattr(requests, "post", lambda *a, **k: Resp())
    result = run_disease_identifier(b"abc", request_id="req-provisional")
    assert result["status"] == "provisional"
    assert result["disease_status"] == "provisional"
    assert result["displayable"] is True
    assert result["uncertain"] is True
    assert result["display_name"] == "Eczema"


def test_disease_identifier_returns_unavailable_without_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = run_disease_identifier(b"abc")
    assert result["status"] == "unavailable"
    assert result["uncertain"] is True


def test_endpoint_runs_pipeline_when_gate_accepts(monkeypatch):
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

    resp = _post_analyze_image(img_bytes)
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["image_gate"]["status"] == "accepted"
    assert body["ml_analysis"]["predicted_class"] == "mild"


def test_endpoint_prefers_gemini_disease_identifier(monkeypatch):
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
        "app.ml.image_model.run_disease_identifier",
        lambda *a, **k: {
            "status": "classified",
            "disease_status": "classified",
            "source": "gemini",
            "predicted_class": "eczema",
            "confidence": 0.77,
            "display_name": "Eczema",
            "canonical_label": "Eczema Photos",
            "top_predictions": [{"label": "eczema", "score": 0.77}],
            "differential_diagnoses": [{"label": "psoriasis"}],
            "all_probabilities": {},
            "uncertain": False,
            "ambiguous": False,
        },
    )
    monkeypatch.setattr(
        "app.ml.image_model._predict_severity",
        lambda data: (
            {"predicted_class": "moderate", "confidence": 0.72, "all_probabilities": {"mild": 0.1, "moderate": 0.72, "severe": 0.18}},
            None,
        ),
    )
    monkeypatch.setattr(
        "app.ml.image_model._predict_disease",
        lambda data: pytest.fail("Legacy disease model should not run when Gemini disease result is available"),
    )
    monkeypatch.setattr("app.ml.image_model.is_image_blurry", lambda img: False)

    resp = _post_analyze_image(img_bytes)
    assert resp.status_code == 200
    body = resp.json()
    assert body["ml_analysis"]["disease"]["source"] == "gemini"
    assert body["ml_analysis"]["disease_display"] == "Eczema"


def test_endpoint_displays_provisional_gemini_disease(monkeypatch):
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
        "app.ml.image_model.run_disease_identifier",
        lambda *a, **k: {
            "status": "provisional",
            "disease_status": "provisional",
            "source": "gemini",
            "predicted_class": "dermatitis",
            "confidence": 0.38,
            "display_name": "Eczema",
            "canonical_label": "Eczema Photos",
            "top_predictions": [{"label": "dermatitis", "score": 0.38}],
            "differential_diagnoses": [{"label": "eczema"}],
            "all_probabilities": {},
            "uncertain": True,
            "displayable": True,
            "ambiguous": True,
        },
    )
    monkeypatch.setattr(
        "app.ml.image_model._predict_severity",
        lambda data: (
            {"predicted_class": "moderate", "confidence": 0.72, "all_probabilities": {"mild": 0.1, "moderate": 0.72, "severe": 0.18}},
            None,
        ),
    )
    monkeypatch.setattr(
        "app.ml.image_model._predict_disease",
        lambda data: pytest.fail("Legacy disease model should not run for provisional Gemini disease result"),
    )
    monkeypatch.setattr("app.ml.image_model.is_image_blurry", lambda img: False)

    resp = _post_analyze_image(img_bytes)
    assert resp.status_code == 200
    body = resp.json()
    assert body["ml_analysis"]["disease"]["disease_status"] == "provisional"
    assert body["ml_analysis"]["disease_display"] == "Eczema"
    assert body["ml_analysis"]["should_suppress_disease_display"] is False


def test_endpoint_skips_ml_when_gate_rejects(monkeypatch):
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

    resp = _post_analyze_image(img_bytes)
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is False
    assert body["image_gate"]["status"] == "rejected"
    assert body["ml_analysis"] is None


def test_severity_uncertain_rule(monkeypatch):
    img_bytes = _make_image_bytes()
    stub_gate = {
        "enabled": True,
        "status": "accepted",
        "accepted": True,
        "model_id": "gemini-stub",
        "latency_ms": 2000,
        "top_label": "rash_like_skin",
        "top_score": 0.9,
        "scores": [],
        "reasons": [],
        "thresholds": {},
    }
    monkeypatch.setattr("app.ml.image_model.run_image_gate", lambda *args, **kwargs: stub_gate)
    monkeypatch.setattr(
        "app.ml.image_model._predict_severity",
        lambda data: (
            {
                "predicted_class": "mild",
                "confidence": 0.35,
                "all_probabilities": {"mild": 0.35, "moderate": 0.33, "severe": 0.32},
            },
            None,
        ),
    )
    monkeypatch.setattr(
        "app.ml.image_model._predict_disease",
        lambda data: {"status": "classified", "predicted_class": "eczema", "confidence": 0.8, "top_predictions": [], "all_probabilities": {}},
    )
    monkeypatch.setattr("app.ml.image_model.is_image_blurry", lambda img: False)

    resp = _post_analyze_image(img_bytes)
    assert resp.status_code == 200
    body = resp.json()
    assert body["ml_analysis"]["severity_uncertain"] is True
    assert body["ml_analysis"]["severity_status"] == "uncertain"


def test_slow_gate_flag_via_endpoint(monkeypatch):
    img_bytes = _make_image_bytes()
    stub_gate = {
        "enabled": True,
        "status": "accepted",
        "accepted": True,
        "model_id": "gemini-stub",
        "latency_ms": 9000,
        "slow_gate": True,
        "performance_warning": "Gemini gate latency high",
        "top_label": "rash_like_skin",
        "top_score": 0.9,
        "scores": [],
        "reasons": [],
        "thresholds": {},
    }
    monkeypatch.setattr("app.ml.image_model.run_image_gate", lambda *args, **kwargs: stub_gate)
    monkeypatch.setattr(
        "app.ml.image_model._predict_severity",
        lambda data: (
            {
                "predicted_class": "mild",
                "confidence": 0.9,
                "all_probabilities": {"mild": 0.9, "moderate": 0.05, "severe": 0.05},
            },
            None,
        ),
    )
    monkeypatch.setattr(
        "app.ml.image_model._predict_disease",
        lambda data: {"status": "classified", "predicted_class": "eczema", "confidence": 0.8, "top_predictions": [], "all_probabilities": {}},
    )
    monkeypatch.setattr("app.ml.image_model.is_image_blurry", lambda img: False)

    resp = _post_analyze_image(img_bytes)
    assert resp.status_code == 200
    body = resp.json()
    assert body["image_gate"]["slow_gate"] is True
    assert "performance_warning" in body["image_gate"]


def test_disease_uncertain_and_ambiguous(monkeypatch):
    img_bytes = _make_image_bytes()
    stub_gate = {
        "enabled": True,
        "status": "accepted",
        "accepted": True,
        "model_id": "gemini-stub",
        "latency_ms": 2000,
        "top_label": "rash_like_skin",
        "top_score": 0.6,
        "scores": [],
        "reasons": [],
        "thresholds": {},
    }
    monkeypatch.setattr("app.ml.image_model.run_image_gate", lambda *args, **kwargs: stub_gate)
    monkeypatch.setattr(
        "app.ml.image_model._predict_severity",
        lambda data: (
            {
                "predicted_class": "moderate",
                "confidence": 0.6,
                "all_probabilities": {"mild": 0.2, "moderate": 0.6, "severe": 0.2},
            },
            None,
        ),
    )
    monkeypatch.setattr(
        "app.ml.image_model._predict_disease",
        lambda data: {
            "status": "classified",
            "predicted_class": "eczema",
            "confidence": 0.54,
            "top_predictions": [
                {"label": "eczema", "score": 0.54},
                {"label": "urticaria", "score": 0.5},
                {"label": "acne", "score": 0.3},
            ],
            "all_probabilities": {},
        },
    )
    monkeypatch.setattr("app.ml.image_model.is_image_blurry", lambda img: False)

    resp = _post_analyze_image(img_bytes)
    body = resp.json()
    disease = body["ml_analysis"]["disease"]
    assert disease["disease_status"] == "uncertain"
    assert disease["ambiguous"] is True


def test_quality_poor_triggers_retake(monkeypatch):
    img_bytes = _make_image_bytes()
    stub_gate = {
        "enabled": True,
        "status": "accepted",
        "accepted": True,
        "model_id": "gemini-stub",
        "latency_ms": 1500,
        "top_label": "clear_or_normal_skin",
        "top_score": 0.7,
        "scores": [],
        "reasons": [],
        "thresholds": {},
    }
    monkeypatch.setattr("app.ml.image_model.run_image_gate", lambda *args, **kwargs: stub_gate)
    monkeypatch.setattr(
        "app.ml.image_model.assess_image_quality",
        lambda img: {"quality_status": "poor", "quality_score": 0.3, "quality_flags": ["blurry"], "retake_required": True},
    )
    monkeypatch.setattr(
        "app.ml.image_model._predict_severity",
        lambda data: (
            {
                "predicted_class": "mild",
                "confidence": 0.9,
                "all_probabilities": {"mild": 0.9, "moderate": 0.05, "severe": 0.05},
            },
            None,
        ),
    )
    monkeypatch.setattr(
        "app.ml.image_model._predict_disease",
        lambda data: {"status": "classified", "predicted_class": "eczema", "confidence": 0.8, "top_predictions": [], "all_probabilities": {}},
    )
    monkeypatch.setattr("app.ml.image_model.is_image_blurry", lambda img: False)

    resp = _post_analyze_image(img_bytes)
    body = resp.json()
    quality = body["image_gate"]["quality"]
    assert quality["quality_status"] == "poor"
    assert body["ml_analysis"]["patient_guidance"]["retake_required"] is True


def test_triage_priority_for_rash_gate(monkeypatch):
    img_bytes = _make_image_bytes()
    stub_gate = {
        "enabled": True,
        "status": "accepted",
        "accepted": True,
        "model_id": "gemini-stub",
        "latency_ms": 1000,
        "top_label": "rash_like_skin",
        "top_score": 0.9,
        "scores": [],
        "reasons": [],
        "thresholds": {},
    }
    monkeypatch.setattr("app.ml.image_model.run_image_gate", lambda *args, **kwargs: stub_gate)
    monkeypatch.setattr(
        "app.ml.image_model._predict_severity",
        lambda data: (
            {
                "predicted_class": "moderate",
                "confidence": 0.7,
                "all_probabilities": {"mild": 0.2, "moderate": 0.7, "severe": 0.1},
            },
            None,
        ),
    )
    monkeypatch.setattr(
        "app.ml.image_model._predict_disease",
        lambda data: {"status": "classified", "predicted_class": "eczema", "confidence": 0.8, "top_predictions": [], "all_probabilities": {}},
    )
    monkeypatch.setattr("app.ml.image_model.is_image_blurry", lambda img: False)

    resp = _post_analyze_image(img_bytes)
    triage = resp.json()["ml_analysis"]["triage"]
    assert triage["triage_level"] == "priority_review"
    assert triage["needs_clinician_review"] is True


def test_triage_urgent_high_severity(monkeypatch):
    img_bytes = _make_image_bytes()
    stub_gate = {
        "enabled": True,
        "status": "accepted",
        "accepted": True,
        "model_id": "gemini-stub",
        "latency_ms": 1200,
        "top_label": "rash_like_skin",
        "top_score": 0.9,
        "scores": [],
        "reasons": [],
        "thresholds": {},
    }
    monkeypatch.setattr("app.ml.image_model.run_image_gate", lambda *args, **kwargs: stub_gate)
    monkeypatch.setattr(
        "app.ml.image_model._predict_severity",
        lambda data: (
            {
                "predicted_class": "severe",
                "confidence": 0.85,
                "all_probabilities": {"mild": 0.05, "moderate": 0.1, "severe": 0.85},
            },
            None,
        ),
    )
    monkeypatch.setattr(
        "app.ml.image_model._predict_disease",
        lambda data: {"status": "classified", "predicted_class": "eczema", "confidence": 0.8, "top_predictions": [], "all_probabilities": {}},
    )
    monkeypatch.setattr("app.ml.image_model.is_image_blurry", lambda img: False)

    resp = _post_analyze_image(img_bytes)
    triage = resp.json()["ml_analysis"]["triage"]
    assert triage["triage_level"] == "urgent_attention"
    assert "high_severity_confidence" in triage["red_flags"]

def test_clear_skin_gate_suppresses_image_analysis(monkeypatch):
    img_bytes = _make_image_bytes()
    stub_gate = {
        "enabled": True,
        "status": "accepted",
        "accepted": True,
        "model_id": "gemini-stub",
        "latency_ms": 800,
        "top_label": "clear_or_normal_skin",
        "top_score": 0.95,
        "scores": [],
        "reasons": [],
        "thresholds": {},
    }
    monkeypatch.setattr("app.ml.image_model.run_image_gate", lambda *args, **kwargs: stub_gate)
    monkeypatch.setattr("app.ml.image_model.is_image_blurry", lambda img: False)

    resp = _post_analyze_image(img_bytes)
    body = resp.json()["ml_analysis"]
    assert body["image_analysis_suppressed"] is True
    assert body["disease_display"] == "Uncertain"
    assert body["severity_display"] == "Uncertain"
    assert body["should_suppress_disease_display"] is True
    assert body["should_suppress_hard_severity"] is True
    assert body["patient_guidance"]["urgent_warning"] is False
    assert body["triage"]["triage_level"] == "self_care"


def test_clear_skin_score_085_suppresses(monkeypatch):
    """Hard suppression should trigger at the lowered 0.85 threshold."""
    img_bytes = _make_image_bytes()
    stub_gate = {
        "enabled": True,
        "status": "accepted",
        "accepted": True,
        "model_id": "gemini-stub",
        "latency_ms": 500,
        "top_label": "clear_or_normal_skin",
        "top_score": 0.85,
        "scores": [],
        "reasons": [],
        "thresholds": {},
    }
    monkeypatch.setattr("app.ml.image_model.run_image_gate", lambda *args, **kwargs: stub_gate)
    monkeypatch.setattr("app.ml.image_model.is_image_blurry", lambda img: False)

    resp = _post_analyze_image(img_bytes)
    ml = resp.json()["ml_analysis"]
    assert ml["image_analysis_suppressed"] is True
    assert ml["suppression_reason"] == "No obvious rash detected in image"
    assert ml["image_assessment_display"] == "No obvious rash detected"
    assert ml["patient_guidance"]["urgent_warning"] is False
    assert ml["patient_guidance"]["summary"].startswith("No obvious rash detected")


def test_clear_skin_score_086_suppresses(monkeypatch):
    img_bytes = _make_image_bytes()
    stub_gate = {
        "enabled": True,
        "status": "accepted",
        "accepted": True,
        "model_id": "gemini-stub",
        "latency_ms": 400,
        "top_label": "clear_or_normal_skin",
        "top_score": 0.86,
        "scores": [],
        "reasons": [],
        "thresholds": {},
    }
    monkeypatch.setattr("app.ml.image_model.run_image_gate", lambda *args, **kwargs: stub_gate)
    monkeypatch.setattr("app.ml.image_model.is_image_blurry", lambda img: False)

    resp = _post_analyze_image(img_bytes)
    ml = resp.json()["ml_analysis"]
    assert ml["image_analysis_suppressed"] is True
    assert ml["should_suppress_disease_display"] is True
    assert ml["should_suppress_hard_severity"] is True
    assert ml["image_assessment_display"] == "No obvious rash detected"


def test_clear_skin_084_runs_models_but_uncertain_display(monkeypatch):
    """Below threshold, models run; displays remain uncertain and gate text is kept."""
    img_bytes = _make_image_bytes()
    stub_gate = {
        "enabled": True,
        "status": "accepted",
        "accepted": True,
        "model_id": "gemini-stub",
        "latency_ms": 700,
        "top_label": "clear_or_normal_skin",
        "top_score": 0.84,
        "scores": [],
        "reasons": [],
        "thresholds": {},
    }
    monkeypatch.setattr("app.ml.image_model.run_image_gate", lambda *args, **kwargs: stub_gate)
    monkeypatch.setattr("app.ml.image_model.is_image_blurry", lambda img: False)
    monkeypatch.setattr(
        "app.ml.image_model._predict_severity",
        lambda data: (
            {
                "predicted_class": "mild",
                "confidence": 0.33,
                "all_probabilities": {"mild": 0.33, "moderate": 0.34, "severe": 0.33},
            },
            None,
        ),
    )
    monkeypatch.setattr(
        "app.ml.image_model._predict_disease",
        lambda data: {
            "status": "classified",
            "predicted_class": "eczema",
            "confidence": 0.5,
            "top_predictions": [{"label": "eczema", "score": 0.5}, {"label": "urticaria", "score": 0.45}],
            "all_probabilities": {"eczema": 0.5, "urticaria": 0.45},
        },
    )

    resp = _post_analyze_image(img_bytes)
    ml = resp.json()["ml_analysis"]
    assert ml["image_analysis_suppressed"] is False
    assert ml["disease_display"] == "Uncertain"
    assert ml["severity_display"] == "Uncertain"
    assert ml["image_assessment_display"] == "clear or normal skin"


def test_visible_rash_not_suppressed(monkeypatch):
    img_bytes = _make_image_bytes()
    stub_gate = {
        "enabled": True,
        "status": "accepted",
        "accepted": True,
        "model_id": "gemini-stub",
        "latency_ms": 600,
        "top_label": "rash_like_skin",
        "top_score": 0.92,
        "scores": [],
        "reasons": [],
        "thresholds": {},
    }
    monkeypatch.setattr("app.ml.image_model.run_image_gate", lambda *args, **kwargs: stub_gate)
    monkeypatch.setattr(
        "app.ml.image_model._predict_severity",
        lambda data: (
            {
                "predicted_class": "moderate",
                "confidence": 0.8,
                "all_probabilities": {"mild": 0.1, "moderate": 0.8, "severe": 0.1},
            },
            None,
        ),
    )
    monkeypatch.setattr(
        "app.ml.image_model._predict_disease",
        lambda data: {
            "status": "classified",
            "predicted_class": "eczema",
            "confidence": 0.82,
            "top_predictions": [{"label": "eczema", "score": 0.82}, {"label": "urticaria", "score": 0.12}],
            "all_probabilities": {"eczema": 0.82, "urticaria": 0.12},
        },
    )
    monkeypatch.setattr("app.ml.image_model.is_image_blurry", lambda img: False)

    resp = _post_analyze_image(img_bytes)
    body = resp.json()["ml_analysis"]
    assert body["image_analysis_suppressed"] is False
    assert body["disease_display"] != "Uncertain"
    assert body["should_suppress_disease_display"] is False
    assert body["severity_display"] == "moderate"
    assert body["should_suppress_hard_severity"] is False


def test_rash_like_image_no_suppression_after_threshold_change(monkeypatch):
    """Ensure rash-like images remain unsuppressed with updated clear-skin gate."""
    img_bytes = _make_image_bytes()
    stub_gate = {
        "enabled": True,
        "status": "accepted",
        "accepted": True,
        "model_id": "gemini-stub",
        "latency_ms": 650,
        "top_label": "rash_like_skin",
        "top_score": 0.88,
        "scores": [],
        "reasons": [],
        "thresholds": {},
    }
    monkeypatch.setattr("app.ml.image_model.run_image_gate", lambda *args, **kwargs: stub_gate)
    monkeypatch.setattr(
        "app.ml.image_model._predict_severity",
        lambda data: (
            {
                "predicted_class": "moderate",
                "confidence": 0.77,
                "all_probabilities": {"mild": 0.1, "moderate": 0.77, "severe": 0.13},
            },
            None,
        ),
    )
    monkeypatch.setattr(
        "app.ml.image_model._predict_disease",
        lambda data: {
            "status": "classified",
            "predicted_class": "eczema",
            "confidence": 0.81,
            "top_predictions": [{"label": "eczema", "score": 0.81}],
            "all_probabilities": {"eczema": 0.81},
        },
    )
    monkeypatch.setattr("app.ml.image_model.is_image_blurry", lambda img: False)

    resp = _post_analyze_image(img_bytes)
    ml = resp.json()["ml_analysis"]
    assert ml["image_analysis_suppressed"] is False
    assert ml["image_assessment_display"] == "Possible rash detected"


def test_low_confidence_severity_sets_uncertain_display(monkeypatch):
    img_bytes = _make_image_bytes()
    stub_gate = {
        "enabled": True,
        "status": "accepted",
        "accepted": True,
        "model_id": "gemini-stub",
        "latency_ms": 700,
        "top_label": "rash_like_skin",
        "top_score": 0.91,
        "scores": [],
        "reasons": [],
        "thresholds": {},
    }
    monkeypatch.setattr("app.ml.image_model.run_image_gate", lambda *args, **kwargs: stub_gate)
    monkeypatch.setattr(
        "app.ml.image_model._predict_severity",
        lambda data: (
            {
                "predicted_class": "mild",
                "confidence": 0.45,
                "all_probabilities": {"mild": 0.45, "moderate": 0.42, "severe": 0.13},
            },
            None,
        ),
    )
    monkeypatch.setattr(
        "app.ml.image_model._predict_disease",
        lambda data: {
            "status": "classified",
            "predicted_class": "urticaria",
            "confidence": 0.7,
            "top_predictions": [{"label": "urticaria", "score": 0.7}],
            "all_probabilities": {"urticaria": 0.7},
        },
    )
    monkeypatch.setattr("app.ml.image_model.is_image_blurry", lambda img: False)

    resp = _post_analyze_image(img_bytes)
    body = resp.json()["ml_analysis"]
    assert body["image_analysis_suppressed"] is False
    assert body["severity_display"] == "Uncertain"
    assert body["should_suppress_hard_severity"] is True

def test_synonym_normalization_ringworm(monkeypatch):
    from app.ml import disease_taxonomy
    canon, display = disease_taxonomy.normalize_condition("ringworm")
    assert canon == "Tinea Ringworm Candidiasis and other Fungal Infections"
    assert display == "Fungal Infection"


def test_synonym_normalization_hives(monkeypatch):
    from app.ml import disease_taxonomy
    canon, display = disease_taxonomy.normalize_condition("hives")
    assert canon == "Urticaria Hives"
    assert display == "Hives (Urticaria)"


def test_synonym_normalization_dermatitis(monkeypatch):
    from app.ml import disease_taxonomy
    canon, display = disease_taxonomy.normalize_condition("dermatitis")
    assert canon == "Eczema Photos"
    assert display == "Eczema"


def test_melanoma_nevi_normalization(monkeypatch):
    from app.ml import disease_taxonomy
    for term in ("melanoma", "mole", "nevi"):
        canon, display = disease_taxonomy.normalize_condition(term)
        assert canon == "Melanoma Skin Cancer Nevi and Moles"
        assert display == "Melanoma / Nevi"


def test_fungal_normalization(monkeypatch):
    from app.ml import disease_taxonomy
    canon, display = disease_taxonomy.normalize_condition("fungal")
    assert canon == "Tinea Ringworm Candidiasis and other Fungal Infections"
    assert display == "Fungal Infection"


def test_top_predictions_enriched(monkeypatch):
    from app.ml.disease_taxonomy import normalize_predictions
    preds = [{"label": "fungal", "score": 0.8}, {"label": "eczema", "score": 0.7}]
    enriched = normalize_predictions(preds)
    assert all("canonical_label" in p and "display_name" in p for p in enriched)
