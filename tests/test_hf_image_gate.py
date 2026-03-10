import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.services import hf_image_gate
from app.services.hf_image_gate import (
    GateThresholds,
    evaluate_gate_scores,
    run_image_gate,
    gate_metrics_snapshot,
    reset_gate_metrics,
)
from app.services.hf_gate_tuning import ThresholdSet, evaluate_threshold_sets


def _make_image_bytes(color=(220, 180, 180)) -> bytes:
    image = Image.new("RGB", (256, 256), color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_gate_accepts_skin_default_thresholds():
    thresholds = GateThresholds()
    scores = [
        {"label": "a close-up photo of human skin", "score": 0.6},
        {"label": "a close-up photo of a skin rash or lesion", "score": 0.4},
        {"label": "a non-skin object", "score": 0.05},
    ]
    decision = evaluate_gate_scores(scores, thresholds)
    assert decision.accepted is True
    assert decision.reasons == []


def test_gate_rejects_non_skin():
    thresholds = GateThresholds(max_non_skin_score=0.2)
    scores = [
        {"label": "a non-skin object", "score": 0.6},
        {"label": "a close-up photo of human skin", "score": 0.3},
        {"label": "a close-up photo of normal healthy skin", "score": 0.2},
    ]
    decision = evaluate_gate_scores(scores, thresholds)
    assert decision.accepted is False
    assert any("non-skin" in r for r in decision.reasons)


def test_gate_rejects_screenshot():
    thresholds = GateThresholds(max_screenshot_score=0.25)
    scores = [
        {"label": "a screenshot or text document", "score": 0.5},
        {"label": "a close-up photo of human skin", "score": 0.4},
    ]
    decision = evaluate_gate_scores(scores, thresholds)
    assert decision.accepted is False
    assert any("screenshot" in r for r in decision.reasons)


def test_gate_rejects_blurry():
    thresholds = GateThresholds(max_blurry_score=0.2)
    scores = [
        {"label": "a blurry medical photo", "score": 0.4},
        {"label": "a close-up photo of human skin", "score": 0.5},
        {"label": "a close-up photo of normal healthy skin", "score": 0.3},
    ]
    decision = evaluate_gate_scores(scores, thresholds)
    assert decision.accepted is False
    assert any("blurry" in r for r in decision.reasons)


def test_reason_codes_are_machine_friendly():
    thresholds = GateThresholds(max_non_skin_score=0.2, max_screenshot_score=0.1, max_blurry_score=0.1)
    scores = [
        {"label": "a non-skin object", "score": 0.9},
        {"label": "a screenshot or text document", "score": 0.5},
        {"label": "a blurry medical photo", "score": 0.3},
        {"label": "a close-up photo of human skin", "score": 0.05},
    ]
    decision = evaluate_gate_scores(scores, thresholds)
    assert decision.reason_codes == ["NON_SKIN_HIGH", "SCREENSHOT_HIGH", "BLURRY_HIGH", "SKIN_LOW"]


def test_fail_open_on_timeout(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "dummy-token")
    monkeypatch.setenv("HF_GATE_FAIL_OPEN", "true")

    class TimeoutClient:
        def zero_shot_image_classification(self, *args, **kwargs):
            raise hf_image_gate.InferenceTimeoutError("timeout")

    monkeypatch.setattr(hf_image_gate, "_get_client", lambda cfg: TimeoutClient())
    result = run_image_gate(b"abc", filename="photo.png", content_type="image/png", request_id="req-timeout")
    assert result["status"] == "fail_open"
    assert result["accepted"] is True
    assert "timeout" in result["reasons"][0].lower()
    assert "TIMEOUT" in result["reason_codes"]


def test_gate_disabled(monkeypatch):
    monkeypatch.setenv("HF_GATE_ENABLED", "false")
    result = run_image_gate(b"abc")
    assert result["status"] == "disabled"
    assert result["accepted"] is True


def test_threshold_env_overrides(monkeypatch):
    monkeypatch.setenv("HF_GATE_MIN_SKIN_SCORE", "0.9")
    monkeypatch.setenv("HF_GATE_MAX_NON_SKIN_SCORE", "0.1")
    monkeypatch.setenv("HF_GATE_MAX_SCREENSHOT_SCORE", "0.05")
    monkeypatch.setenv("HF_GATE_MAX_BLURRY_SCORE", "0.2")
    monkeypatch.setenv("HF_GATE_MIN_RASH_OR_NORMAL_SCORE", "0.8")
    cfg = hf_image_gate.load_gate_config()
    assert cfg.thresholds.min_skin_score == 0.9
    assert cfg.thresholds.max_non_skin_score == 0.1
    assert cfg.thresholds.max_screenshot_score == 0.05
    assert cfg.thresholds.max_blurry_score == 0.2
    assert cfg.thresholds.min_rash_or_normal_score == 0.8


def test_endpoint_runs_pipeline_when_gate_accepts(monkeypatch):
    client = TestClient(app)
    img_bytes = _make_image_bytes()

    stub_gate = {
        "enabled": True,
        "status": "accepted",
        "accepted": True,
        "model_id": "stub-model",
        "latency_ms": 12.3,
        "top_label": "a close-up photo of human skin",
        "top_score": 0.92,
        "scores": [],
        "reasons": [],
        "thresholds": {},
    }

    monkeypatch.setattr("app.ml.image_model.run_image_gate", lambda *args, **kwargs: stub_gate)
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

    response = client.post(
        "/ml/analyze-image",
        files={"file": ("photo.png", img_bytes, "image/png")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["image_gate"]["status"] == "accepted"
    assert payload["ml_analysis"]["predicted_class"] == "mild"
    assert payload["ml_analysis"]["disease"]["predicted_class"] == "eczema"


def test_endpoint_skips_ml_when_gate_rejects(monkeypatch):
    client = TestClient(app)
    img_bytes = _make_image_bytes()

    stub_gate = {
        "enabled": True,
        "status": "rejected",
        "accepted": False,
        "model_id": "stub-model",
        "latency_ms": 10,
        "top_label": "a non-skin object",
        "top_score": 0.8,
        "scores": [],
        "reasons": ["non-skin score too high"],
        "thresholds": {},
    }

    monkeypatch.setattr("app.ml.image_model.run_image_gate", lambda *args, **kwargs: stub_gate)
    monkeypatch.setattr("app.ml.image_model.is_image_blurry", lambda img: False)

    response = client.post(
        "/ml/analyze-image",
        files={"file": ("photo.png", img_bytes, "image/png")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is False
    assert payload["image_gate"]["status"] == "rejected"
    assert payload["ml_analysis"] is None
    assert "clear" in payload["message"].lower()


def test_recent_decision_buffer_bounded():
    reset_gate_metrics()
    thresholds = {"min_skin_score": 0.3}
    for i in range(60):
        hf_image_gate.gate_metrics.record_decision(
            status="rejected",
            reason_codes=["NON_SKIN_HIGH"],
            top_label="a non-skin object",
            latency_ms=100 + i,
            content_type="image/png",
            size_bytes=10_000,
            thresholds=thresholds,
            decision_meta={
                "timestamp": i,
                "request_id": f"req-{i}",
                "status": "rejected",
                "top_label": "a non-skin object",
                "top_score": 0.9,
                "reasons": ["x"],
                "reason_codes": ["NON_SKIN_HIGH"],
                "latency_ms": 100 + i,
                "image_size_bytes": 10_000,
                "content_type": "image/png",
                "filename": f"file-{i}.png",
            },
        )
    snap = gate_metrics_snapshot()
    assert len(snap["recent_decisions"]) == 50
    assert snap["recent_decisions"][0]["request_id"] == "req-10"


def test_top_label_metrics_accumulate():
    reset_gate_metrics()
    thresholds = {"min_skin_score": 0.3}
    hf_image_gate.gate_metrics.record_decision(
        status="accepted",
        reason_codes=[],
        top_label="a close-up photo of human skin",
        latency_ms=120,
        content_type="image/jpeg",
        size_bytes=200000,
        thresholds=thresholds,
        decision_meta={},
    )
    hf_image_gate.gate_metrics.record_decision(
        status="accepted",
        reason_codes=[],
        top_label="a close-up photo of human skin",
        latency_ms=90,
        content_type="image/jpeg",
        size_bytes=200000,
        thresholds=thresholds,
        decision_meta={},
    )
    hf_image_gate.gate_metrics.record_decision(
        status="rejected",
        reason_codes=["NON_SKIN_HIGH"],
        top_label="a non-skin object",
        latency_ms=110,
        content_type="image/png",
        size_bytes=50000,
        thresholds=thresholds,
        decision_meta={},
    )
    snap = gate_metrics_snapshot()
    assert snap["top_labels"]["a close-up photo of human skin"] == 2
    assert snap["top_labels"]["a non-skin object"] == 1


def test_threshold_tuning_ranks_sets(tmp_path):
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            [
                {"image_path": "img1.png", "expected_gate_label": "accepted"},
                {"image_path": "img2.png", "expected_gate_label": "rejected"},
            ]
        )
    )

    # Stub scorer: first image skin-like, second non-skin
    def fake_infer_scores(path: str):
        if "img1" in path:
            scores = [
                {"label": "a close-up photo of human skin", "score": 0.8},
                {"label": "a non-skin object", "score": 0.1},
            ]
        else:
            scores = [
                {"label": "a non-skin object", "score": 0.7},
                {"label": "a close-up photo of human skin", "score": 0.2},
            ]
        return scores, "accepted"

    sets = [
        ThresholdSet(name="strict", thresholds=GateThresholds(max_non_skin_score=0.2)),
        ThresholdSet(name="loose", thresholds=GateThresholds(max_non_skin_score=0.8)),
    ]

    summary = evaluate_threshold_sets(str(dataset_path), sets, infer_scores_fn=fake_infer_scores)
    ranked = summary["ranked_results"]
    assert ranked[0]["name"] == "strict"  # strict should cut non-skin
    assert ranked[0]["counts"]["false_accept"] == 0


def test_model_metrics_includes_dashboard(monkeypatch):
    client = TestClient(app)
    # ensure snapshot has known structure
    reset_gate_metrics()
    hf_image_gate.gate_metrics.record_decision(
        status="accepted",
        reason_codes=[],
        top_label="a close-up photo of human skin",
        latency_ms=50,
        content_type="image/png",
        size_bytes=50_000,
        thresholds={"min_skin_score": 0.3},
        decision_meta={},
    )

    res = client.get("/ml/model-metrics")
    assert res.status_code == 200
    body = res.json()
    assert "image_gate_metrics" in body
    ig = body["image_gate_metrics"]
    assert "recent_decisions" in ig
    assert "reason_counts" in ig
