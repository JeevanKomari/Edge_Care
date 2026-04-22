import os
from datetime import datetime, timezone


def health_check():
    return {
        "status": "OK",
        "service": "edgecare-ml",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "avg_latency_ms": 420,
        "disease_identifier": "gemini_primary",
        "legacy_disease_fallback_enabled": os.getenv("LEGACY_DISEASE_FALLBACK_ENABLED", "true").lower() in ("1", "true", "yes", "y"),
    }
