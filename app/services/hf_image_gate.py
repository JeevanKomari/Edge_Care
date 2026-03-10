import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from huggingface_hub import InferenceClient
from huggingface_hub.errors import HTTPError

try:
    from huggingface_hub.inference._client import InferenceTimeoutError  # type: ignore
except Exception:  # pragma: no cover - fallback for older hub versions
    class InferenceTimeoutError(TimeoutError):
        """Fallback timeout error when huggingface_hub doesn't expose the class."""


logger = logging.getLogger(__name__)

CANDIDATE_LABELS = [
    "a close-up photo of human skin",
    "a close-up photo of a skin rash or lesion",
    "a close-up photo of normal healthy skin",
    "a blurry medical photo",
    "a screenshot or text document",
    "a non-skin object",
]


@dataclass
class GateThresholds:
    min_skin_score: float = 0.35
    max_non_skin_score: float = 0.35
    max_screenshot_score: float = 0.30
    max_blurry_score: float = 0.35
    min_rash_or_normal_score: float = 0.25


@dataclass
class GateConfig:
    enabled: bool
    token: Optional[str]
    model_id: str
    timeout: float
    fail_open: bool
    thresholds: GateThresholds


@dataclass
class GateDecision:
    accepted: bool
    reasons: List[str]
    reason_codes: List[str]
    normalized_scores: List[Dict[str, Any]]
    top_label: Optional[str]
    top_score: Optional[float]


class GateMetrics:
    def __init__(self) -> None:
        self._counters: Dict[str, int] = {
            "gate_requests_total": 0,
            "gate_accept_total": 0,
            "gate_reject_total": 0,
            "gate_error_total": 0,
            "gate_fail_open_total": 0,
            "gate_timeout_total": 0,
        }
        self._status_counts: Dict[str, int] = {
            "accepted": 0,
            "rejected": 0,
            "fail_open": 0,
            "error": 0,
            "disabled": 0,
        }
        self._reason_counts: Dict[str, int] = {}
        self._top_label_counts: Dict[str, int] = {}
        self._content_type_counts: Dict[str, int] = {}
        self._size_buckets: Dict[str, int] = {
            "lt_100kb": 0,
            "100kb_500kb": 0,
            "500kb_1mb": 0,
            "gt_1mb": 0,
        }
        self._latency_buckets: Dict[str, int] = {
            "lt_250ms": 0,
            "250ms_750ms": 0,
            "750ms_1500ms": 0,
            "gt_1500ms": 0,
        }
        self._latencies: List[float] = []
        self._recent: List[Dict[str, Any]] = []
        self._recent_max = 50
        self._last_thresholds: Optional[Dict[str, Any]] = None
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            for k in self._counters:
                self._counters[k] = 0
            for k in self._status_counts:
                self._status_counts[k] = 0
            self._reason_counts.clear()
            self._top_label_counts.clear()
            self._content_type_counts.clear()
            for k in self._size_buckets:
                self._size_buckets[k] = 0
            for k in self._latency_buckets:
                self._latency_buckets[k] = 0
            self._latencies.clear()
            self._recent.clear()
            self._last_thresholds = None

    def inc(self, key: str) -> None:
        with self._lock:
            if key not in self._counters:
                self._counters[key] = 0
            self._counters[key] += 1

    def observe_latency(self, latency_ms: float) -> None:
        with self._lock:
            self._latencies.append(latency_ms)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            counters_copy = dict(self._counters)
            latencies = list(self._latencies)
            status_counts = dict(self._status_counts)
            reason_counts = dict(self._reason_counts)
            top_labels = dict(self._top_label_counts)
            content_types = dict(self._content_type_counts)
            size_buckets = dict(self._size_buckets)
            latency_buckets = dict(self._latency_buckets)
            recent = list(self._recent)
            last_thresholds = dict(self._last_thresholds) if self._last_thresholds else None
        if latencies:
            sorted_latencies = sorted(latencies)
            p95_index = max(int(len(sorted_latencies) * 0.95) - 1, 0)
            latency_stats = {
                "count": len(sorted_latencies),
                "avg_ms": round(sum(sorted_latencies) / len(sorted_latencies), 2),
                "p95_ms": round(sorted_latencies[p95_index], 2),
                "max_ms": round(max(sorted_latencies), 2),
                "min_ms": round(min(sorted_latencies), 2),
            }
        else:
            latency_stats = {
                "count": 0,
                "avg_ms": 0.0,
                "p95_ms": 0.0,
                "max_ms": 0.0,
                "min_ms": 0.0,
            }
        return {
            "counters": counters_copy,
            "status_counts": status_counts,
            "reason_counts": reason_counts,
            "top_labels": top_labels,
            "content_types": content_types,
            "size_buckets": size_buckets,
            "latency_buckets": latency_buckets,
            "gate_latency_ms": latency_stats,
            "recent_decisions": recent,
            "threshold_snapshot": last_thresholds,
            **counters_copy,
        }

    def record_decision(
        self,
        status: str,
        reason_codes: List[str],
        top_label: Optional[str],
        latency_ms: float,
        content_type: Optional[str],
        size_bytes: Optional[int],
        thresholds: Dict[str, Any],
        decision_meta: Dict[str, Any],
    ) -> None:
        with self._lock:
            # status counter
            if status in self._status_counts:
                self._status_counts[status] += 1
            else:
                self._status_counts[status] = 1

            # reason counters
            if not reason_codes:
                self._reason_counts.setdefault("none", 0)
                self._reason_counts["none"] += 1
            else:
                for code in reason_codes:
                    self._reason_counts[code] = self._reason_counts.get(code, 0) + 1
                if len(reason_codes) > 1:
                    self._reason_counts["MULTIPLE_REASONS"] = self._reason_counts.get("MULTIPLE_REASONS", 0) + 1

            # top label counts
            if top_label:
                self._top_label_counts[top_label] = self._top_label_counts.get(top_label, 0) + 1

            # content type counts
            if content_type:
                self._content_type_counts[content_type] = self._content_type_counts.get(content_type, 0) + 1

            # size buckets
            if size_bytes is not None:
                if size_bytes < 100 * 1024:
                    bucket = "lt_100kb"
                elif size_bytes < 500 * 1024:
                    bucket = "100kb_500kb"
                elif size_bytes < 1024 * 1024:
                    bucket = "500kb_1mb"
                else:
                    bucket = "gt_1mb"
                self._size_buckets[bucket] += 1

            # latency buckets
            if latency_ms is not None:
                if latency_ms < 250:
                    lb = "lt_250ms"
                elif latency_ms < 750:
                    lb = "250ms_750ms"
                elif latency_ms < 1500:
                    lb = "750ms_1500ms"
                else:
                    lb = "gt_1500ms"
                self._latency_buckets[lb] += 1
                self._latencies.append(latency_ms)

            # recent buffer
            self._recent.append(decision_meta)
            if len(self._recent) > self._recent_max:
                self._recent = self._recent[-self._recent_max :]

            # last thresholds snapshot
            self._last_thresholds = thresholds


gate_metrics = GateMetrics()
_client_cache: Dict[Tuple[str, bool], InferenceClient] = {}


def _str_to_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def load_gate_config() -> GateConfig:
    thresholds = GateThresholds(
        min_skin_score=_get_env_float("HF_GATE_MIN_SKIN_SCORE", GateThresholds.min_skin_score),
        max_non_skin_score=_get_env_float("HF_GATE_MAX_NON_SKIN_SCORE", GateThresholds.max_non_skin_score),
        max_screenshot_score=_get_env_float("HF_GATE_MAX_SCREENSHOT_SCORE", GateThresholds.max_screenshot_score),
        max_blurry_score=_get_env_float("HF_GATE_MAX_BLURRY_SCORE", GateThresholds.max_blurry_score),
        min_rash_or_normal_score=_get_env_float("HF_GATE_MIN_RASH_OR_NORMAL_SCORE", GateThresholds.min_rash_or_normal_score),
    )

    model_id = os.getenv("HF_MODEL_ID") or "openai/clip-vit-large-patch14-336"
    enabled = _str_to_bool(os.getenv("HF_GATE_ENABLED"), True)
    token = os.getenv("HF_TOKEN")
    timeout = _get_env_float("HF_GATE_TIMEOUT", 8.0)
    fail_open = _str_to_bool(os.getenv("HF_GATE_FAIL_OPEN"), True)

    return GateConfig(
        enabled=enabled,
        token=token,
        model_id=model_id,
        timeout=timeout,
        fail_open=fail_open,
        thresholds=thresholds,
    )


def _get_client(cfg: GateConfig) -> InferenceClient:
    key = (cfg.model_id, bool(cfg.token))
    if key not in _client_cache:
        _client_cache[key] = InferenceClient(model=cfg.model_id, token=cfg.token)
    return _client_cache[key]


def normalize_scores(raw_scores: Any) -> List[Dict[str, Any]]:
    normalized = []
    if isinstance(raw_scores, list):
        for item in raw_scores:
            if not isinstance(item, dict):
                continue
            label = item.get("label") or item.get("class")
            score_val = item.get("score") or item.get("probability")
            try:
                score = float(score_val)
            except (TypeError, ValueError):
                continue
            if label:
                normalized.append({"label": str(label), "score": score})
    normalized.sort(key=lambda x: x.get("score", 0), reverse=True)
    return normalized


def evaluate_gate_scores(scores: List[Dict[str, Any]], thresholds: GateThresholds) -> GateDecision:
    normalized = normalize_scores(scores)
    score_map = {item["label"]: float(item.get("score", 0.0)) for item in normalized}
    reasons: List[str] = []
    reason_codes: List[str] = []

    skin_score = score_map.get("a close-up photo of human skin", 0.0)
    rash_score = score_map.get("a close-up photo of a skin rash or lesion", 0.0)
    normal_score = score_map.get("a close-up photo of normal healthy skin", 0.0)
    non_skin_score = score_map.get("a non-skin object", 0.0)
    screenshot_score = score_map.get("a screenshot or text document", 0.0)
    blurry_score = score_map.get("a blurry medical photo", 0.0)

    if non_skin_score > thresholds.max_non_skin_score:
        reasons.append(f"non-skin score {non_skin_score:.3f} above {thresholds.max_non_skin_score}")
        reason_codes.append("NON_SKIN_HIGH")
    if screenshot_score > thresholds.max_screenshot_score:
        reasons.append(f"screenshot score {screenshot_score:.3f} above {thresholds.max_screenshot_score}")
        reason_codes.append("SCREENSHOT_HIGH")
    if blurry_score > thresholds.max_blurry_score:
        reasons.append(f"blurry score {blurry_score:.3f} above {thresholds.max_blurry_score}")
        reason_codes.append("BLURRY_HIGH")
    if skin_score < thresholds.min_skin_score:
        reasons.append(f"skin score {skin_score:.3f} below {thresholds.min_skin_score}")
        reason_codes.append("SKIN_LOW")

    rash_or_normal = max(rash_score, normal_score)
    if rash_or_normal < thresholds.min_rash_or_normal_score:
        reasons.append(
            f"rash/normal score {rash_or_normal:.3f} below {thresholds.min_rash_or_normal_score}"
        )
        reason_codes.append("RASH_OR_NORMAL_LOW")

    accepted = len(reasons) == 0
    top_label = normalized[0]["label"] if normalized else None
    top_score = float(normalized[0]["score"]) if normalized else None

    return GateDecision(
        accepted=accepted,
        reasons=reasons,
        reason_codes=reason_codes,
        normalized_scores=normalized,
        top_label=top_label,
        top_score=top_score,
    )


def run_image_gate(
    image_bytes: bytes,
    filename: Optional[str] = None,
    content_type: Optional[str] = None,
    request_id: Optional[str] = None,
    record_metrics: bool = True,
) -> Dict[str, Any]:
    cfg = load_gate_config()
    gate_metrics.inc("gate_requests_total")

    response: Dict[str, Any] = {
        "enabled": cfg.enabled,
        "model_id": cfg.model_id,
        "accepted": True,
        "status": "disabled" if not cfg.enabled else "accepted",
        "latency_ms": 0,
        "top_label": None,
        "top_score": None,
        "scores": [],
        "reasons": [],
        "reason_codes": [],
        "thresholds": {
            "min_skin_score": cfg.thresholds.min_skin_score,
            "max_non_skin_score": cfg.thresholds.max_non_skin_score,
            "max_screenshot_score": cfg.thresholds.max_screenshot_score,
            "max_blurry_score": cfg.thresholds.max_blurry_score,
            "min_rash_or_normal_score": cfg.thresholds.min_rash_or_normal_score,
        },
    }

    if not cfg.enabled:
        _log_gate_event(response, filename, content_type, len(image_bytes), request_id)
        if record_metrics:
            gate_metrics.record_decision(
                status=response["status"],
                reason_codes=response["reason_codes"],
                top_label=response["top_label"],
                latency_ms=response["latency_ms"],
                content_type=content_type,
                size_bytes=len(image_bytes),
                thresholds=response["thresholds"],
                decision_meta=_recent_meta(response, filename, content_type, len(image_bytes), request_id),
            )
        return response

    if not cfg.token:
        response.update({
            "accepted": cfg.fail_open,
            "status": "fail_open" if cfg.fail_open else "error",
            "reasons": ["HF token missing"],
            "reason_codes": ["TOKEN_MISSING"],
        })
        _record_error_metrics(response["status"])
        _log_gate_event(response, filename, content_type, len(image_bytes), request_id)
        if record_metrics:
            gate_metrics.record_decision(
                status=response["status"],
                reason_codes=response["reason_codes"],
                top_label=response["top_label"],
                latency_ms=response["latency_ms"],
                content_type=content_type,
                size_bytes=len(image_bytes),
                thresholds=response["thresholds"],
                decision_meta=_recent_meta(response, filename, content_type, len(image_bytes), request_id),
            )
        return response

    try:
        client = _get_client(cfg)
        start = time.perf_counter()
        raw_scores = client.zero_shot_image_classification(
            image=image_bytes,
            candidate_labels=CANDIDATE_LABELS,
            timeout=cfg.timeout,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        response["latency_ms"] = round(latency_ms, 2)

        decision = evaluate_gate_scores(raw_scores, cfg.thresholds)
        if not decision.normalized_scores:
            raise ValueError("HF gate returned empty scores")
        response.update({
            "accepted": decision.accepted,
            "status": "accepted" if decision.accepted else "rejected",
            "scores": decision.normalized_scores,
            "top_label": decision.top_label,
            "top_score": decision.top_score,
            "reasons": decision.reasons,
            "reason_codes": decision.reason_codes,
        })

        if decision.accepted:
            gate_metrics.inc("gate_accept_total")
        else:
            gate_metrics.inc("gate_reject_total")

    except InferenceTimeoutError as exc:  # timeout path
        response.update({
            "accepted": cfg.fail_open,
            "status": "fail_open" if cfg.fail_open else "error",
            "reasons": ["HF gate timeout"],
            "reason_codes": ["TIMEOUT"],
        })
        gate_metrics.inc("gate_timeout_total")
        _record_error_metrics(response["status"])
        logger.warning("HF image gate timeout", extra={"request_id": request_id})
    except HTTPError as exc:
        response.update({
            "accepted": cfg.fail_open,
            "status": "fail_open" if cfg.fail_open else "error",
            "reasons": [f"HF HTTP error: {exc.response.status_code if hasattr(exc, 'response') else exc}"],
            "reason_codes": ["HTTP_ERROR"],
        })
        _record_error_metrics(response["status"])
        logger.warning("HF image gate HTTP error", extra={"request_id": request_id})
    except Exception as exc:
        reason = str(exc) or exc.__class__.__name__
        response.update({
            "accepted": cfg.fail_open,
            "status": "fail_open" if cfg.fail_open else "error",
            "reasons": [f"HF gate failed: {reason}"],
            "reason_codes": ["GATE_EXCEPTION"],
        })
        _record_error_metrics(response["status"])
        logger.exception("HF image gate failure", extra={"request_id": request_id})

    _log_gate_event(response, filename, content_type, len(image_bytes), request_id)
    if record_metrics:
        gate_metrics.record_decision(
            status=response["status"],
            reason_codes=response.get("reason_codes", []),
            top_label=response.get("top_label"),
            latency_ms=response.get("latency_ms", 0.0) or 0.0,
            content_type=content_type,
            size_bytes=len(image_bytes),
            thresholds=response.get("thresholds", {}),
            decision_meta=_recent_meta(response, filename, content_type, len(image_bytes), request_id),
        )
    return response


def _record_error_metrics(status: str) -> None:
    gate_metrics.inc("gate_error_total")
    if status == "fail_open":
        gate_metrics.inc("gate_fail_open_total")


def _log_gate_event(
    gate_response: Dict[str, Any],
    filename: Optional[str],
    content_type: Optional[str],
    size_bytes: Optional[int],
    request_id: Optional[str],
) -> None:
    payload = {
        "request_id": request_id or uuid.uuid4().hex,
        "filename": filename,
        "content_type": content_type,
        "image_size_bytes": size_bytes,
        "hf_model_id": gate_response.get("model_id"),
        "hf_latency_ms": gate_response.get("latency_ms"),
        "gate_status": gate_response.get("status"),
        "top_label": gate_response.get("top_label"),
        "top_score": gate_response.get("top_score"),
        "thresholds": gate_response.get("thresholds"),
        "reasons": gate_response.get("reasons"),
        "reason_codes": gate_response.get("reason_codes"),
    }

    log_fn = logger.info if gate_response.get("status") == "accepted" else logger.warning
    log_fn("hf_image_gate", extra={"gate": payload})


def gate_metrics_snapshot() -> Dict[str, Any]:
    return gate_metrics.snapshot()


def reset_gate_metrics() -> None:
    """Reset gate metrics (intended for tests/admin)."""
    gate_metrics.reset()


def _recent_meta(
    gate_response: Dict[str, Any],
    filename: Optional[str],
    content_type: Optional[str],
    size_bytes: Optional[int],
    request_id: Optional[str],
) -> Dict[str, Any]:
    return {
        "timestamp": time.time(),
        "request_id": request_id or uuid.uuid4().hex,
        "status": gate_response.get("status"),
        "top_label": gate_response.get("top_label"),
        "top_score": gate_response.get("top_score"),
        "reasons": gate_response.get("reasons"),
        "reason_codes": gate_response.get("reason_codes"),
        "latency_ms": gate_response.get("latency_ms"),
        "image_size_bytes": size_bytes,
        "content_type": content_type,
        "filename": filename,
    }

