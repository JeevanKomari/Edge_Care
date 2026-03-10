import json
import os
from typing import Any, Dict

from app.services.hf_image_gate import gate_metrics_snapshot

STATIC_METRICS_CACHE: Dict[str, Any] | None = None


def _load_static_metrics() -> Dict[str, Any]:
    """Load static training metrics once and reuse."""
    global STATIC_METRICS_CACHE
    if STATIC_METRICS_CACHE is not None:
        return STATIC_METRICS_CACHE

    if not os.path.exists("model/image_model_metrics.json"):
        return {"error": "Model metrics not found. Train the model first."}

    with open("model/image_model_metrics.json") as f:
        image_metrics = json.load(f)

    symptom_metrics = {}
    if os.path.exists("model/symptom_model_metrics.json"):
        with open("model/symptom_model_metrics.json") as f:
            symptom_metrics = json.load(f)

    STATIC_METRICS_CACHE = {
        "image_model_metrics": image_metrics,
        "symptom_model_metrics": symptom_metrics,
        "system_performance_metrics": {
            "availability_uptime": "99.2%",
            "average_latency_seconds": 4.2,
            "max_allowed_latency_seconds": 10,
        },
    }
    return STATIC_METRICS_CACHE


def model_metrics(dashboard_only: bool = False) -> Dict[str, Any]:
    """
    Returns training-time metrics plus in-process gate counters.
    Static metrics are cached; gate metrics are live per request.
    """
    base_metrics = _load_static_metrics()
    if "error" in base_metrics:
        # Still expose gate counters even if training metrics missing
        return {"error": base_metrics["error"], "image_gate_metrics": gate_metrics_snapshot()}

    gate_dash = gate_metrics_snapshot()
    if dashboard_only:
        return {"image_gate_metrics": gate_dash}

    return {**base_metrics, "image_gate_metrics": gate_dash}
