import base64
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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
    connect_timeout: float
    read_timeout: float
    max_retries: int
    fail_open: bool
    debug: bool


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
        connect_timeout=float(os.getenv("GEMINI_GATE_CONNECT_TIMEOUT", 10)),
        read_timeout=float(os.getenv("GEMINI_GATE_READ_TIMEOUT", 15)),
        max_retries=int(os.getenv("GEMINI_GATE_MAX_RETRIES", 2)),
        fail_open=_str_to_bool(os.getenv("GEMINI_GATE_FAIL_OPEN"), True),
        debug=_str_to_bool(os.getenv("GEMINI_GATE_DEBUG"), False),
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
            "gate_retry_total": 0,
        }
        self._latencies: List[float] = []
        self._label_counts: Dict[str, int] = {}
        self._model_usage: Dict[str, int] = {}
        self._reason_code_counts: Dict[str, int] = {}
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

    def record_reason_code(self, code: str) -> None:
        with self._lock:
            self._reason_code_counts[code] = self._reason_code_counts.get(code, 0) + 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            counters = dict(self._counters)
            lat = list(self._latencies)
            labels = dict(self._label_counts)
            models = dict(self._model_usage)
            reason_codes = dict(self._reason_code_counts)
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
            "reason_code_counts": reason_codes,
            **counters,
        }

    def reset(self) -> None:
        with self._lock:
            for k in self._counters:
                self._counters[k] = 0
            self._latencies.clear()
            self._label_counts.clear()
            self._model_usage.clear()
            self._reason_code_counts.clear()


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


def _reason_code_for_http(status: int) -> str:
    if status == 400:
        return "GATE_HTTP_400"
    if status == 401:
        return "GATE_HTTP_401"
    if status == 403:
        return "GATE_HTTP_403"
    if status == 404:
        return "GATE_HTTP_404"
    if status == 429:
        return "GATE_HTTP_429"
    if status == 500:
        return "GATE_HTTP_500"
    if status == 503:
        return "GATE_HTTP_503"
    if status == 502:
        return "GATE_HTTP_502"
    if status == 504:
        return "GATE_HTTP_504"
    return "GATE_HTTP_OTHER"


def run_image_gate(image_bytes: bytes, filename: Optional[str] = None, content_type: Optional[str] = None, request_id: Optional[str] = None) -> Dict[str, Any]:
    cfg = load_gate_config()
    request_id = request_id or uuid.uuid4().hex
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
        response.update({"accepted": cfg.fail_open, "status": "fail_open" if cfg.fail_open else "error", "reasons": ["Gemini API key missing"], "reason_codes": ["GATE_PROVIDER_ERROR"]})
        metrics.record_reason_code("GATE_PROVIDER_ERROR")
        metrics.inc("gate_fail_open_total" if cfg.fail_open else "gate_error_total")
        return response

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{cfg.model}:generateContent"
    payload = _build_payload(image_bytes)
    headers = {"Content-Type": "application/json"}
    params = {"key": cfg.api_key}
    timeouts: Tuple[float, float] = (cfg.connect_timeout, cfg.read_timeout)

    def log_safe(level, msg, **extra_fields):
        safe = {
            "request_id": request_id,
            "provider": "gemini",
            "model_id": cfg.model,
            "endpoint": url,
            "has_api_key": bool(cfg.api_key),
            "connect_timeout": cfg.connect_timeout,
            "read_timeout": cfg.read_timeout,
            "image_size_bytes": len(image_bytes) if image_bytes is not None else None,
            "content_type": content_type,
            "prompt_mode": "json",
            **extra_fields,
        }
        if cfg.debug:
            logger.log(level, msg, extra={"gate": safe})
        else:
            filtered = {k: v for k, v in safe.items() if k not in {"body_preview"}}
            logger.log(level, msg, extra={"gate": filtered})

    try:
        attempt = 0
        start_total = time.perf_counter()
        last_exc: Optional[Exception] = None
        while attempt <= cfg.max_retries:
            attempt += 1
            attempt_start = time.perf_counter()
            try:
                log_safe(logging.INFO, "gemini_request_start", attempt=attempt)
                resp = requests.post(url, headers=headers, params=params, json=payload, timeout=timeouts)
                latency_ms = (time.perf_counter() - attempt_start) * 1000
                response["latency_ms"] = round((time.perf_counter() - start_total) * 1000, 2)
                metrics.observe_latency(response["latency_ms"])

                status = resp.status_code
                body_preview = resp.text[:1000] if cfg.debug else resp.text[:500]
                log_safe(
                    logging.INFO,
                    "gemini_response",
                    attempt=attempt,
                    status_code=status,
                    latency_ms=response["latency_ms"],
                    body_preview=body_preview,
                )

                if status in {502, 503, 504} and attempt <= cfg.max_retries:
                    metrics.inc("gate_retry_total")
                    time.sleep(min(0.5 * (2 ** (attempt - 1)), 2.0))
                    last_exc = requests.HTTPError(f"HTTP {status}")
                    continue

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
                    metrics.record_reason_code("GATE_EMPTY_RESPONSE")
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
                log_safe(logging.INFO, "gemini_parse_success", attempt=attempt, label=label, confidence=confidence)
                if label in {"rash_like_skin", "clear_or_normal_skin"}:
                    response["accepted"] = True
                    response["status"] = "accepted"
                    metrics.inc("gate_accept_total")
                else:
                    response["accepted"] = False
                    response["status"] = "rejected"
                    metrics.inc("gate_reject_total")
                break
            except requests.Timeout as exc:
                last_exc = exc
                metrics.inc("gate_timeout_total")
                metrics.record_reason_code("GATE_TIMEOUT")
                response["latency_ms"] = round((time.perf_counter() - start_total) * 1000, 2)
                log_safe(logging.WARNING, "gemini_timeout", attempt=attempt, latency_ms=response["latency_ms"], error=str(exc))
                if attempt > cfg.max_retries:
                    raise
                metrics.inc("gate_retry_total")
                time.sleep(min(0.5 * (2 ** (attempt - 1)), 2.0))
                continue
            except requests.RequestException as exc:
                last_exc = exc
                response["latency_ms"] = round((time.perf_counter() - start_total) * 1000, 2)
                rc = _reason_code_for_http(getattr(exc.response, "status_code", None) or 0) if hasattr(exc, "response") else "GATE_NETWORK_ERROR"
                metrics.record_reason_code(rc)
                log_safe(logging.WARNING, "gemini_http_error", attempt=attempt, status_code=getattr(exc.response, "status_code", None), latency_ms=response["latency_ms"], error=str(exc))
                if getattr(exc.response, "status_code", 0) in {502, 503, 504} and attempt <= cfg.max_retries:
                    metrics.inc("gate_retry_total")
                    time.sleep(min(0.5 * (2 ** (attempt - 1)), 2.0))
                    continue
                raise
            except ValueError as exc:
                last_exc = exc
                response["latency_ms"] = round((time.perf_counter() - start_total) * 1000, 2)
                metrics.record_reason_code("GATE_JSON_PARSE_ERROR")
                log_safe(logging.WARNING, "gemini_parse_error", attempt=attempt, latency_ms=response["latency_ms"], error=str(exc))
                raise

        else:
            if last_exc:
                raise last_exc

    except requests.Timeout as exc:
        response.update({"accepted": cfg.fail_open, "status": "fail_open" if cfg.fail_open else "error", "reasons": ["Gemini gate timeout"], "reason_codes": ["GATE_TIMEOUT"]})
        if cfg.fail_open:
            metrics.inc("gate_fail_open_total")
        else:
            metrics.inc("gate_error_total")
        log_safe(logging.WARNING, "gemini_timeout_final", error=str(exc), latency_ms=response.get("latency_ms"))
    except requests.RequestException as exc:
        status = getattr(exc.response, "status_code", None)
        rc = _reason_code_for_http(status or 0) if status is not None else "GATE_NETWORK_ERROR"
        response.update({"accepted": cfg.fail_open, "status": "fail_open" if cfg.fail_open else "error", "reasons": [f"Gemini gate HTTP error: {status or exc}"], "reason_codes": [rc]})
        if cfg.fail_open:
            metrics.inc("gate_fail_open_total")
        else:
            metrics.inc("gate_error_total")
        log_safe(logging.WARNING, "gemini_http_error_final", status_code=status, error=str(exc), latency_ms=response.get("latency_ms"))
    except ValueError as exc:
        response.update({"accepted": cfg.fail_open, "status": "fail_open" if cfg.fail_open else "error", "reasons": [f"Gemini response invalid: {exc}"], "reason_codes": ["GATE_JSON_PARSE_ERROR"]})
        if cfg.fail_open:
            metrics.inc("gate_fail_open_total")
        else:
            metrics.inc("gate_error_total")
        log_safe(logging.WARNING, "gemini_parse_error_final", error=str(exc), latency_ms=response.get("latency_ms"))
    except Exception as exc:
        response.update({"accepted": cfg.fail_open, "status": "fail_open" if cfg.fail_open else "error", "reasons": [f"Gemini gate failed: {exc}"], "reason_codes": ["GATE_PROVIDER_ERROR"]})
        metrics.inc("gate_error_total")
        if cfg.fail_open:
            metrics.inc("gate_fail_open_total")
        log_safe(logging.ERROR, "gemini_exception", error=str(exc), latency_ms=response.get("latency_ms"))
        logger.exception("Gemini image gate failure", extra={"request_id": request_id})

    return response


def gate_metrics_snapshot() -> Dict[str, Any]:
    return metrics.snapshot()


def reset_gate_metrics() -> None:
    metrics.reset()
