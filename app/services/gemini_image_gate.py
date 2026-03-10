import base64
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

GATE_LABELS = {
    "rash_like_skin",
    "clear_or_normal_skin",
    "non_skin_or_irrelevant_image",
    "unclear_or_low_quality",
}


@dataclass
class GateConfig:
    enabled: bool
    api_key: Optional[str]
    model: str
    timeout: float
    fail_open: bool


def _str_to_bool(val: Optional[str], default: bool = False) -> bool:
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_gate_config() -> GateConfig:
    return GateConfig(
        enabled=_str_to_bool(os.getenv("GEMINI_GATE_ENABLED"), True),
        api_key=os.getenv("GEMINI_API_KEY"),
        model=os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"),
        timeout=float(os.getenv("GEMINI_GATE_TIMEOUT", 15)),
        fail_open=_str_to_bool(os.getenv("GEMINI_GATE_FAIL_OPEN"), True),
    )


class GateMetrics:
    def __init__(self) -> None:
        self._counters = {
            "gate_requests_total": 0,
            "gate_accept_total": 0,
            "gate_reject_total": 0,
            "gate_error_total": 0,
            "gate_fail_open_total": 0,
            "gate_timeout_total": 0,
        }
        self._latencies: List[float] = []
        self._label_counts: Dict[str, int] = {}
        self._model_usage: Dict[str, int] = {}
        self._lock = threading.Lock()

    def inc(self, key: str) -> None:
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + 1

    def observe_latency(self, latency_ms: float) -> None:
        with self._lock:
            self._latencies.append(latency_ms)

    def record_label(self, label: str) -> None:
        with self._lock:
            self._label_counts[label] = self._label_counts.get(label, 0) + 1

    def record_model(self, model: str) -> None:
        with self._lock:
            self._model_usage[model] = self._model_usage.get(model, 0) + 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            counters = dict(self._counters)
            lat = list(self._latencies)
            labels = dict(self._label_counts)
            models = dict(self._model_usage)
        if lat:
            lat_sorted = sorted(lat)
            p95_idx = max(int(len(lat_sorted) * 0.95) - 1, 0)
            latency_stats = {
                "count": len(lat_sorted),
                "avg_ms": round(sum(lat_sorted) / len(lat_sorted), 2),
                "p95_ms": round(lat_sorted[p95_idx], 2),
                "max_ms": round(max(lat_sorted), 2),
                "min_ms": round(min(lat_sorted), 2),
            }
        else:
            latency_stats = {"count": 0, "avg_ms": 0, "p95_ms": 0, "max_ms": 0, "min_ms": 0}
        return {
            "counters": counters,
            "gate_latency_ms": latency_stats,
            "label_counts": labels,
            "model_usage": models,
            **counters,
        }

    def reset(self) -> None:
        with self._lock:
            for k in self._counters:
                self._counters[k] = 0
            self._latencies.clear()
            self._label_counts.clear()
            self._model_usage.clear()


metrics = GateMetrics()


def _build_payload(image_bytes: bytes) -> Dict[str, Any]:
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return {
        "contents": [
            {
                "parts": [
                    {
                        "text": (
                            "You are an image gate. Classify the image into one of: "
                            "rash_like_skin, clear_or_normal_skin, non_skin_or_irrelevant_image, unclear_or_low_quality. "
                            "Respond ONLY with JSON: {\"label\":...,\"confidence\":...,\"reason\":...}."
                        )
                    },
                    {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
        },
    }


def _parse_response(text: str) -> Dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from Gemini: {e}")
    label = data.get("label")
    confidence = data.get("confidence")
    reason = data.get("reason")
    if label not in GATE_LABELS:
        raise ValueError("Label missing or invalid")
    try:
        conf = float(confidence)
    except Exception:
        conf = 0.0
    return {"label": label, "confidence": conf, "reason": reason or ""}


def run_image_gate(image_bytes: bytes, filename: Optional[str] = None, content_type: Optional[str] = None, request_id: Optional[str] = None) -> Dict[str, Any]:
    cfg = load_gate_config()
    metrics.inc("gate_requests_total")
    response: Dict[str, Any] = {
        "enabled": cfg.enabled,
        "model_id": cfg.model,
        "accepted": True,
        "status": "disabled" if not cfg.enabled else "accepted",
        "latency_ms": 0,
        "top_label": None,
        "top_score": None,
        "scores": [],
        "reasons": [],
        "reason_codes": [],
        "attempted_models": [cfg.model],
        "provider_unavailable_models": [],
        "thresholds": {},
    }

    if not cfg.enabled:
        return response
    if not cfg.api_key:
        response.update({"accepted": cfg.fail_open, "status": "fail_open" if cfg.fail_open else "error", "reasons": ["Gemini API key missing"], "reason_codes": ["GATE_EXCEPTION"]})
        metrics.inc("gate_fail_open_total" if cfg.fail_open else "gate_error_total")
        return response

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{cfg.model}:generateContent?key={cfg.api_key}"
    payload = _build_payload(image_bytes)
    headers = {"Content-Type": "application/json"}

    try:
        start = time.perf_counter()
        resp = requests.post(url, headers=headers, json=payload, timeout=cfg.timeout)
        latency_ms = (time.perf_counter() - start) * 1000
        response["latency_ms"] = round(latency_ms, 2)
        metrics.observe_latency(latency_ms)
        if resp.status_code == 408:
            raise requests.Timeout("Gemini timeout")
        resp.raise_for_status()
        result_json = resp.json()
        text = None
        if isinstance(result_json, dict):
            candidates = result_json.get("candidates") or []
            if candidates and "content" in candidates[0]:
                parts = candidates[0]["content"].get("parts") or []
                if parts and "text" in parts[0]:
                    text = parts[0]["text"]
        if not text:
            raise ValueError("No text content returned")
        parsed = _parse_response(text)
        label = parsed["label"]
        confidence = parsed["confidence"]
        response.update({
            "top_label": label,
            "top_score": confidence,
            "scores": [{"label": label, "score": confidence}],
            "reasons": [parsed.get("reason", "")],
            "reason_codes": [],
            "model_id": cfg.model,
        })
        metrics.record_label(label)
        metrics.record_model(cfg.model)
        if label in {"rash_like_skin", "clear_or_normal_skin"}:
            response["accepted"] = True
            response["status"] = "accepted"
            metrics.inc("gate_accept_total")
        else:
            response["accepted"] = False
            response["status"] = "rejected"
            metrics.inc("gate_reject_total")
    except requests.Timeout:
        response.update({"accepted": cfg.fail_open, "status": "fail_open" if cfg.fail_open else "error", "reasons": ["Gemini gate timeout"], "reason_codes": ["GATE_TIMEOUT"]})
        metrics.inc("gate_timeout_total")
        if cfg.fail_open:
            metrics.inc("gate_fail_open_total")
        if not cfg.fail_open:
            metrics.inc("gate_error_total")
    except Exception as exc:
        response.update({"accepted": cfg.fail_open, "status": "fail_open" if cfg.fail_open else "error", "reasons": [f"Gemini gate failed: {exc}"], "reason_codes": ["GATE_EXCEPTION"]})
        metrics.inc("gate_error_total")
        if cfg.fail_open:
            metrics.inc("gate_fail_open_total")
        logger.exception("Gemini image gate failure", extra={"request_id": request_id})

    return response


def gate_metrics_snapshot() -> Dict[str, Any]:
    return metrics.snapshot()


def reset_gate_metrics() -> None:
    metrics.reset()
