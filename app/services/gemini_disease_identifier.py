import base64
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests

from app.ml.disease_taxonomy import normalize_condition


@dataclass
class DiseaseIdentifierConfig:
    enabled: bool
    api_key: Optional[str]
    model: str
    connect_timeout: float
    read_timeout: float
    confidence_threshold: float
    provisional_threshold: float


def _str_to_bool(val: Optional[str], default: bool = False) -> bool:
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_disease_identifier_config() -> DiseaseIdentifierConfig:
    return DiseaseIdentifierConfig(
        enabled=_str_to_bool(os.getenv("GEMINI_DISEASE_IDENTIFIER_ENABLED"), True),
        api_key=os.getenv("GEMINI_API_KEY"),
        model=os.getenv("GEMINI_DISEASE_MODEL", os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")),
        connect_timeout=float(os.getenv("GEMINI_DISEASE_CONNECT_TIMEOUT", 8)),
        read_timeout=float(os.getenv("GEMINI_DISEASE_READ_TIMEOUT", 12)),
        confidence_threshold=float(os.getenv("GEMINI_DISEASE_CONFIDENCE_THRESHOLD", 0.45)),
        provisional_threshold=float(os.getenv("GEMINI_DISEASE_PROVISIONAL_THRESHOLD", 0.35)),
    )


def _build_payload(image_bytes: bytes) -> Dict[str, Any]:
    b64 = base64.b64encode(image_bytes).decode("ascii")
    prompt = (
        "You are assisting a skin-triage app. Review the uploaded image and respond ONLY as JSON with keys "
        "\"condition\", \"confidence\", \"differentials\", \"reason\", and \"visible_skin_concern\". "
        "If a visible rash, lesion, discoloration, swelling, acne-like eruption, scaling, hives, wart, mole, or infection pattern is present, "
        "return the best short visual condition name even when the exact diagnosis is uncertain. "
        "Prefer concrete skin terms such as eczema, dermatitis, psoriasis, hives, fungal infection, acne, rosacea, wart, cellulitis, impetigo, pigmentation disorder, mole, or benign lesion when visually appropriate. "
        "Use null only when the image is non-skin, normal/clear skin, too blurry, too dark, too close, or no reliable skin concern is visible. "
        "\"confidence\" must be calibrated from 0 to 1: use 0.35-0.44 for a weak but visible provisional match, 0.45-0.69 for a reasonable visual match, and 0.70+ only for a strong visual pattern. "
        "\"differentials\" must be an array with up to 3 short alternative condition names. "
        "\"visible_skin_concern\" must be true or false. "
        "This is a preliminary visual guess, not a confirmed diagnosis."
    )
    return {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
        },
    }


def _normalize_label(label: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if not label:
        return None, None, None
    raw = str(label).strip()
    if not raw:
        return None, None, None
    canonical, display = normalize_condition(raw)
    return raw, canonical, display


def _normalize_differentials(items: Any) -> List[Dict[str, Any]]:
    if not isinstance(items, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for item in items[:3]:
        raw = item.get("label") if isinstance(item, dict) else item
        raw_label, canonical, display = _normalize_label(raw)
        if not raw_label:
            continue
        entry: Dict[str, Any] = {"label": raw_label}
        if canonical:
            entry["canonical_label"] = canonical
        if display:
            entry["display_name"] = display
        normalized.append(entry)
    return normalized


def _parse_response(text: str, confidence_threshold: float, provisional_threshold: float) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    data = json.loads(cleaned)
    condition = data.get("condition")
    confidence_raw = data.get("confidence", 0)
    differentials = _normalize_differentials(data.get("differentials"))
    reason = str(data.get("reason") or "").strip()
    visible = bool(data.get("visible_skin_concern"))

    try:
        confidence = float(confidence_raw)
    except Exception:
        confidence = 0.0

    raw_label, canonical, display = _normalize_label(condition)
    if not visible or not raw_label:
        return {
            "status": "uncertain",
            "disease_status": "uncertain",
            "source": "gemini",
            "predicted_class": None,
            "confidence": round(confidence, 4),
            "display_name": None,
            "canonical_label": None,
            "top_predictions": differentials,
            "differential_diagnoses": differentials,
            "all_probabilities": {},
            "uncertain": True,
            "displayable": False,
            "ambiguous": False,
            "reason": reason or "No reliable visible disease pattern identified.",
        }

    if confidence >= confidence_threshold:
        status = "classified"
    elif confidence >= provisional_threshold:
        status = "provisional"
    else:
        status = "uncertain"
    ambiguous = len(differentials) > 1
    result = {
        "status": status,
        "disease_status": status,
        "source": "gemini",
        "predicted_class": raw_label,
        "confidence": round(confidence, 4),
        "top_predictions": [{"label": raw_label, "score": round(confidence, 4)}],
        "differential_diagnoses": differentials,
        "all_probabilities": {},
        "uncertain": status != "classified",
        "displayable": status in {"classified", "provisional"},
        "ambiguous": ambiguous and status != "classified",
        "reason": reason,
    }
    if canonical:
        result["canonical_label"] = canonical
    if display:
        result["display_name"] = display
        result["predicted_class_display"] = display
    return result


def run_disease_identifier(image_bytes: bytes, request_id: Optional[str] = None) -> Dict[str, Any]:
    cfg = load_disease_identifier_config()
    response: Dict[str, Any] = {
        "enabled": cfg.enabled,
        "model_id": cfg.model,
        "source": "gemini",
        "status": "disabled" if not cfg.enabled else "unavailable",
        "disease_status": "disabled" if not cfg.enabled else "unavailable",
        "predicted_class": None,
        "confidence": 0.0,
        "display_name": None,
        "canonical_label": None,
        "top_predictions": [],
        "differential_diagnoses": [],
        "all_probabilities": {},
        "uncertain": True,
        "displayable": False,
        "ambiguous": False,
        "reason": None,
        "request_id": request_id,
    }

    if not cfg.enabled:
        return response
    if not cfg.api_key:
        response["reason"] = "Gemini API key missing"
        return response

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{cfg.model}:generateContent"
    payload = _build_payload(image_bytes)
    headers = {"Content-Type": "application/json"}
    params = {"key": cfg.api_key}
    timeout: Tuple[float, float] = (cfg.connect_timeout, cfg.read_timeout)

    try:
        resp = requests.post(url, headers=headers, params=params, json=payload, timeout=timeout)
        resp.raise_for_status()
        body = resp.json()
        candidates = body.get("candidates") or []
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        text = parts[0].get("text") if parts else None
        if not text:
            response["reason"] = "No Gemini disease text returned"
            return response
        parsed = _parse_response(text, cfg.confidence_threshold, cfg.provisional_threshold)
        parsed["enabled"] = True
        parsed["model_id"] = cfg.model
        parsed["request_id"] = request_id
        return parsed
    except Exception as exc:
        response["reason"] = f"Gemini disease identifier unavailable: {exc}"
        return response
