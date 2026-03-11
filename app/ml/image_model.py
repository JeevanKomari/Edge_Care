import io
import json
import os
import uuid
import logging
from typing import Dict, Any

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import tensorflow as tf

from app.services.gemini_image_gate import run_image_gate, metrics as gemini_metrics
from app.ml.disease_taxonomy import normalize_condition, normalize_predictions

tflite = tf.lite
logger = logging.getLogger(__name__)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_DIR = os.path.join(BASE_DIR, "model")
EXPORTS_DIR = os.path.join(BASE_DIR, "exports", "edgecare_6class_v1")

# Clear skin gate controls
CLEAR_SKIN_LABEL = "clear_or_normal_skin"
# Hard suppression threshold for obvious clear/normal skin
CLEAR_SKIN_SUPPRESSION_THRESHOLD = float(os.getenv("CLEAR_SKIN_SUPPRESSION_THRESHOLD", "0.85"))
# Display normalization threshold to keep patient wording consistent
CLEAR_SKIN_DISPLAY_THRESHOLD = 0.85
RUN_SUPPRESSED_IMAGE_MODELS = os.getenv("RUN_SUPPRESSED_IMAGE_MODELS", "false").lower() in ("1", "true", "yes", "y")

# ============================================================
# 1) Existing Severity Model (mild/moderate/severe) -- unchanged
# ============================================================
SEVERITY_MODEL_PATH = os.path.join(MODEL_DIR, "skin_model.tflite")
SEVERITY_CLASS_NAMES = ["mild", "moderate", "severe"]

severity_interpreter = None
severity_input_details = None
severity_output_details = None

try:
    if os.path.exists(SEVERITY_MODEL_PATH):
        severity_interpreter = tflite.Interpreter(model_path=SEVERITY_MODEL_PATH)
        severity_interpreter.allocate_tensors()
        severity_input_details = severity_interpreter.get_input_details()
        severity_output_details = severity_interpreter.get_output_details()
    else:
        print("WARN: Severity model not found at:", SEVERITY_MODEL_PATH)
except Exception as e:
    severity_interpreter = None
    severity_input_details = None
    severity_output_details = None
    print(f"WARN: Severity model load failed: {e}")

# ============================================================
# 2) New Disease Model (6-class TFLite)
# ============================================================
DISEASE_MODEL_TFLITE_PATH = os.path.join(EXPORTS_DIR, "edgecare_6class_effnetb0_best.tflite")
DISEASE_CLASS_NAMES_PATH = os.path.join(EXPORTS_DIR, "class_names.json")
DISEASE_TOP_K = 3
UNCERTAIN_TOP1_THRESHOLD = 0.45
UNCERTAIN_MARGIN_THRESHOLD = 0.08

# Toggle for disease preprocessing scale. True = /255 (current), False = raw 0-255
DISEASE_TFLITE_NORMALIZE = os.getenv("DISEASE_TFLITE_NORMALIZE", "true").lower() in ("1", "true", "yes", "y")
DISEASE_TFLITE_DEBUG = os.getenv("DISEASE_TFLITE_DEBUG", "false").lower() in ("1", "true", "yes", "y")

disease_interpreter = None
disease_input_details = None
disease_output_details = None
disease_class_names = None


def _load_disease_class_names():
    global disease_class_names
    if disease_class_names is not None:
        return True
    try:
        with open(DISEASE_CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
            disease_class_names = json.load(f)
        return True
    except Exception as e:
        print(f"WARN: Failed to load disease class names: {e}")
        disease_class_names = None
        return False


def _load_disease_tflite():
    global disease_interpreter, disease_input_details, disease_output_details
    if disease_interpreter is not None:
        return True
    if not os.path.exists(DISEASE_MODEL_TFLITE_PATH):
        print("WARN: Disease model not found at:", DISEASE_MODEL_TFLITE_PATH)
        return False
    try:
        disease_interpreter = tflite.Interpreter(model_path=DISEASE_MODEL_TFLITE_PATH)
        disease_interpreter.allocate_tensors()
        disease_input_details = disease_interpreter.get_input_details()
        disease_output_details = disease_interpreter.get_output_details()
        return True
    except Exception as e:
        disease_interpreter = None
        disease_input_details = None
        disease_output_details = None
        print(f"WARN: Disease model load failed: {e}")
        return False


def _ensure_disease_artifacts():
    model_ok = _load_disease_tflite()
    classes_ok = _load_disease_class_names()
    return model_ok and classes_ok


# ============================================================
# Common helpers
# ============================================================
def preprocess_image_224(image: Image.Image, normalize: bool = True) -> np.ndarray:
    """Resize to 224x224, optional normalize to [0,1], add batch dim."""
    image = image.resize((224, 224))
    arr = np.array(image)
    if normalize:
        arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr.astype(np.float32)


def is_image_blurry(image: Image.Image, threshold: float = 100.0) -> bool:
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold


def assess_image_quality(image: Image.Image) -> Dict[str, Any]:
    """Lightweight heuristic quality assessment."""
    arr = np.array(image)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    mean = gray.mean()
    std = gray.std()
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    center = gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    center_var = center.var()
    edges = cv2.Canny(gray, 50, 150)
    edge_density = edges.mean() / 255.0

    flags = []
    if lap_var < 80:
        flags.append("blurry")
    if mean < 70:
        flags.append("low_brightness")
    if std < 25:
        flags.append("low_contrast")
    if edge_density < 0.02:
        flags.append("too_far")
    if center_var < (0.5 * std if std > 0 else 0):
        flags.append("lesion_not_centered")
    white_ratio = (arr > 245).mean()
    if white_ratio > 0.4:
        flags.append("screenshot_or_ui_elements")

    penalty = 0.0
    for f in flags:
        penalty += 0.15
    quality_score = max(0.0, 1.0 - min(penalty, 0.9))
    if quality_score >= 0.7:
        status = "good"
    elif quality_score >= 0.5:
        status = "borderline"
    else:
        status = "poor"
    return {
        "quality_status": status,
        "quality_score": round(float(quality_score), 3),
        "quality_flags": flags,
        "retake_required": status == "poor",
    }


def _softmax_if_needed(x: np.ndarray) -> np.ndarray:
    """Ensure probability distribution; safe for logits or probs."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1:
        x = x.reshape(-1)
    s = float(np.sum(x))
    if 0.98 <= s <= 1.02 and np.all(x >= 0.0) and np.all(x <= 1.0):
        return x
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


def _predict_severity(input_data: np.ndarray):
    if severity_interpreter is None:
        return None, {"error": "Severity model not found. Please add model/skin_model.tflite"}

    severity_interpreter.set_tensor(severity_input_details[0]["index"], input_data)
    severity_interpreter.invoke()
    pred = severity_interpreter.get_tensor(severity_output_details[0]["index"])[0]
    pred = _softmax_if_needed(pred)

    idx = int(np.argmax(pred))
    label = SEVERITY_CLASS_NAMES[idx]
    conf = float(pred[idx])

    return {
        "predicted_class": label,
        "confidence": round(conf, 4),
        "all_probabilities": {
            "mild": round(float(pred[0]), 4),
            "moderate": round(float(pred[1]), 4),
            "severe": round(float(pred[2]), 4),
        },
    }, None


def _predict_disease(input_data: np.ndarray):
    if not _ensure_disease_artifacts():
        return {
            "status": "disabled",
            "message": (
                "Disease model artifacts missing. "
                "Ensure exports/edgecare_6class_v1 with .tflite and class_names.json is present."
            ),
        }

    if disease_input_details is None or disease_output_details is None:
        return {"status": "disabled", "message": "Disease model not initialized."}

    try:
        # Match interpreter expected dtype/shape
        expected_dtype = disease_input_details[0]["dtype"]
        data = input_data.astype(expected_dtype, copy=False)
        expected_shape = disease_input_details[0]["shape"]
        if list(expected_shape) != list(data.shape):
            try:
                data = data.reshape(expected_shape)
            except Exception:
                return {
                    "status": "disabled",
                    "message": f"Input shape mismatch. Expected {expected_shape}, got {data.shape}.",
                }

        disease_interpreter.set_tensor(disease_input_details[0]["index"], data)
        disease_interpreter.invoke()
        raw = disease_interpreter.get_tensor(disease_output_details[0]["index"])[0]
    except Exception as e:
        return {"status": "disabled", "message": f"Disease model inference failed: {e}"}

    probs = _softmax_if_needed(raw)

    if disease_class_names is None or len(disease_class_names) != len(probs):
        return {
            "status": "disabled",
            "message": "Disease class names missing or length mismatch with model output.",
        }

    sorted_idx = np.argsort(probs)[::-1]
    top1_idx = int(sorted_idx[0])
    top2_idx = int(sorted_idx[1]) if len(sorted_idx) > 1 else top1_idx

    top1_score = float(probs[top1_idx])
    top2_score = float(probs[top2_idx])

    uncertain = (top1_score < UNCERTAIN_TOP1_THRESHOLD) or (
        (top1_score - top2_score) < UNCERTAIN_MARGIN_THRESHOLD
    )

    top_predictions = []
    for idx in sorted_idx[:DISEASE_TOP_K]:
        top_predictions.append(
            {
                "label": disease_class_names[idx],
                "score": round(float(probs[idx]), 4),
            }
        )

    all_probabilities = {
        disease_class_names[i]: round(float(probs[i]), 4) for i in range(len(probs))
    }

    result = {
        "status": "uncertain" if uncertain else "classified",
        "predicted_class": disease_class_names[top1_idx],
        "confidence": round(top1_score, 4),
        "uncertain": uncertain,
        "top_predictions": top_predictions,
        "all_probabilities": all_probabilities,
    }

    if uncertain:
        result["message"] = "Possible skin issue detected, but category is uncertain."

    if DISEASE_TFLITE_DEBUG:
        print(
            "[DISEASE_DEBUG] normalize=",
            DISEASE_TFLITE_NORMALIZE,
            "input_dtype=",
            disease_input_details[0]["dtype"] if disease_input_details else None,
            "input_shape=",
            disease_input_details[0]["shape"] if disease_input_details else None,
            "top_predictions=",
            result.get("top_predictions"),
        )

    return result


# ============================================================
# Public API called by /ml/analyze-image
# ============================================================
def analyze_image(file, file_bytes: bytes | None = None):
    """Run HF image gate first, then existing severity + disease pipeline."""
    request_id = uuid.uuid4().hex
    filename = getattr(file, "filename", None)
    content_type = getattr(file, "content_type", None)

    if file_bytes is None:
        try:
            file_bytes = file.file.read()
        except Exception as e:
            return {"success": False, "error": f"Failed to read upload: {e}"}

    if not file_bytes:
        return {"success": False, "error": "No image bytes received."}

    if content_type and not content_type.startswith("image/"):
        gate_result = {
            "enabled": False,
            "model_id": None,
            "status": "error",
            "accepted": False,
            "reasons": ["Unsupported content type"],
            "latency_ms": 0,
            "top_label": None,
            "top_score": None,
            "scores": [],
            "reason_codes": [],
            "thresholds": {},
        }
        return {
            "success": False,
            "image_gate": gate_result,
            "ml_analysis": None,
            "message": "Unsupported file type. Please upload an image.",
        }

    gate_result = run_image_gate(
        file_bytes,
        filename=filename,
        content_type=content_type,
        request_id=request_id,
    )

    if gate_result.get("latency_ms", 0) > 8000:
        gate_result["slow_gate"] = True
        gate_result["performance_warning"] = "Gemini gate latency high"
        logger.warning(
            "gemini_gate_slow",
            extra={"gate": {"latency_ms": gate_result.get("latency_ms"), "model_id": gate_result.get("model_id")}},
        )
    else:
        gate_result["slow_gate"] = gate_result.get("slow_gate", False)

    gate_status = gate_result.get("status")
    if gate_status == "error":
        return {
            "success": False,
            "image_gate": gate_result,
            "ml_analysis": None,
            "message": "Image gate unavailable. Please retry shortly.",
        }
    if gate_status == "rejected":
        return {
            "success": False,
            "image_gate": gate_result,
            "ml_analysis": None,
            "message": "Image rejected. Please upload a clear, close-up photo of the skin area.",
        }

    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except UnidentifiedImageError:
        return {
            "success": False,
            "image_gate": gate_result,
            "ml_analysis": None,
            "message": "Uploaded file is not a valid image.",
        }
    except Exception as e:
        return {
            "success": False,
            "image_gate": gate_result,
            "ml_analysis": None,
            "message": f"Failed to read image: {e}",
        }

    if is_image_blurry(image):
        return {
            "success": False,
            "image_gate": gate_result,
            "ml_analysis": None,
            "message": "Image quality too low. Please retake the photo.",
        }

    quality_result = assess_image_quality(image)
    if quality_result.get("retake_required"):
        gemini_metrics.inc_retake_required()
        logger.info("quality_poor", extra={"quality": quality_result})
    gate_result["quality"] = quality_result

    gate_top_label = gate_result.get("top_label")
    gate_top_score = gate_result.get("top_score") or 0.0
    clear_skin_suppressed = (gate_top_label == CLEAR_SKIN_LABEL) and (gate_top_score >= CLEAR_SKIN_SUPPRESSION_THRESHOLD)
    suppression_reason = "No obvious rash detected in image" if clear_skin_suppressed else None
    image_assessment_display = "No obvious rash detected" if clear_skin_suppressed else "Image processed"
    should_run_models = (not clear_skin_suppressed) or RUN_SUPPRESSED_IMAGE_MODELS

    # Severity retains original normalization (/255). Disease can toggle via env for mismatch debug.
    severity_input = preprocess_image_224(image, normalize=True) if should_run_models else None
    disease_input = preprocess_image_224(image, normalize=DISEASE_TFLITE_NORMALIZE) if should_run_models else None

    if should_run_models:
        severity_result, severity_err = _predict_severity(severity_input)
        if severity_err:
            return {
                "success": False,
                "image_gate": gate_result,
                "ml_analysis": None,
                "message": severity_err.get("error", "Severity model unavailable"),
            }
    else:
        severity_result = {
            "predicted_class": "suppressed",
            "confidence": 0.0,
            "all_probabilities": {"mild": 0.0, "moderate": 0.0, "severe": 0.0},
            "severity_status": "suppressed",
            "severity_uncertain": True,
            "confidence_band": "low",
            "suppressed_by_gate": True,
        }
        severity_err = None

    # Severity uncertainty rule
    sev_probs = severity_result.get("all_probabilities", {}) if isinstance(severity_result, dict) else {}
    max_conf = max(sev_probs.values()) if sev_probs else 0.0
    sorted_sev = sorted(sev_probs.items(), key=lambda x: x[1], reverse=True)
    sev_top1 = sorted_sev[0][1] if sorted_sev else 0.0
    sev_top2 = sorted_sev[1][1] if len(sorted_sev) > 1 else 0.0
    severity_uncertain = (max_conf < 0.40) or ((sev_top1 - sev_top2) < 0.08)
    severity_status = "uncertain" if severity_uncertain else severity_result.get("predicted_class")
    if sev_top1 >= 0.7:
        sev_band = "high"
    elif sev_top1 >= 0.5:
        sev_band = "medium"
    else:
        sev_band = "low"
    severity_result["severity_status"] = severity_status
    severity_result["severity_uncertain"] = severity_uncertain
    severity_result["confidence_band"] = sev_band
    if severity_uncertain and should_run_models:
        severity_result["warning"] = "Severity confidence is low; classification uncertain."
        gemini_metrics.inc_severity_uncertain()
        logger.info(
            "severity_uncertain_triggered",
            extra={
                "severity": {
                    "predicted_class": severity_result.get("predicted_class"),
                    "max_conf": max_conf,
                    "all_probabilities": sev_probs,
                }
            },
        )

    if should_run_models:
        disease_result = _predict_disease(disease_input)
    else:
        disease_result = {
            "status": "suppressed",
            "predicted_class": "suppressed",
            "confidence": 0.0,
            "uncertain": True,
            "top_predictions": [],
            "all_probabilities": {},
            "suppressed_by_gate": True,
            "message": "Image analysis suppressed by clear-skin gate.",
        }

    # Disease uncertainty / ambiguity
    if disease_result.get("status") not in ("suppressed", "disabled"):
        try:
            top_preds = disease_result.get("top_predictions", [])
            top1 = top_preds[0]["score"] if top_preds else 0.0
            top2 = top_preds[1]["score"] if len(top_preds) > 1 else 0.0
            disease_status = "classified"
            ambiguous = False
            if (top1 < 0.55) or ((top1 - top2) < 0.15):
                disease_status = "uncertain"
                ambiguous = (top1 - top2) < 0.15
                gemini_metrics.inc_disease_uncertain()
            if top1 >= 0.75:
                disease_band = "high"
            elif top1 >= 0.55:
                disease_band = "medium"
            else:
                disease_band = "low"
            disease_result["disease_status"] = disease_status
            disease_result["ambiguous"] = ambiguous
            disease_result["confidence_band"] = disease_band
            disease_result["differential_diagnoses"] = top_preds[:3]
            # Normalize labels to canonical taxonomy for all outputs
            disease_result["top_predictions"] = normalize_predictions(top_preds)
            disease_result["differential_diagnoses"] = normalize_predictions(disease_result["differential_diagnoses"])
            pred_label = disease_result.get("predicted_class")
            canonical_pc, display_pc = normalize_condition(pred_label)
            if canonical_pc:
                disease_result["canonical_label"] = canonical_pc
            if display_pc:
                disease_result["display_name"] = display_pc
            top1_label = top_preds[0]["label"] if top_preds else pred_label
            canonical, display = normalize_condition(top1_label)
            # Prefer canonical/display for primary prediction if available
            if canonical and not disease_result.get("canonical_label"):
                disease_result["canonical_label"] = canonical
            if display and not disease_result.get("display_name"):
                disease_result["display_name"] = display
            if display_pc:
                disease_result["predicted_class_display"] = display_pc
        except Exception:
            pass

    # Normalized display fields
    should_suppress_disease_display = clear_skin_suppressed
    should_suppress_hard_severity = clear_skin_suppressed

    disease_conf = disease_result.get("confidence", 0.0) if isinstance(disease_result, dict) else 0.0
    disease_status = disease_result.get("disease_status") or disease_result.get("status") if isinstance(disease_result, dict) else None
    if disease_status in ("uncertain", "disabled") or disease_result.get("uncertain"):
        should_suppress_disease_display = True
    if disease_conf < 0.55:
        should_suppress_disease_display = True

    severity_conf = severity_result.get("confidence", 0.0) if isinstance(severity_result, dict) else 0.0
    if severity_result.get("severity_uncertain") or severity_conf < 0.5 or ((sev_top1 - sev_top2) < 0.08):
        should_suppress_hard_severity = True

    disease_display = (
        disease_result.get("display_name")
        or disease_result.get("predicted_class_display")
        or disease_result.get("predicted_class")
        or "Uncertain"
    )
    severity_display = severity_result.get("predicted_class") or "Uncertain"

    if should_suppress_disease_display:
        disease_display = "Uncertain"
    if should_suppress_hard_severity:
        severity_display = "Uncertain"

    if not clear_skin_suppressed:
        if gate_top_label == "rash_like_skin":
            image_assessment_display = "Possible rash detected"
        elif gate_top_label:
            image_assessment_display = gate_top_label.replace("_", " ")

    # Normalized benign-clear display even when models run and outputs are uncertain
    if (
        gate_top_label == CLEAR_SKIN_LABEL
        and gate_top_score >= CLEAR_SKIN_DISPLAY_THRESHOLD
        and severity_display == "Uncertain"
        and disease_display == "Uncertain"
    ):
        image_assessment_display = "No obvious rash detected"

    ml_payload = dict(severity_result)
    ml_payload["disease"] = disease_result
    ml_payload["image_analysis_suppressed"] = clear_skin_suppressed
    ml_payload["suppression_reason"] = suppression_reason
    ml_payload["disease_display"] = disease_display
    ml_payload["severity_display"] = severity_display
    ml_payload["should_suppress_disease_display"] = should_suppress_disease_display
    ml_payload["should_suppress_hard_severity"] = should_suppress_hard_severity
    ml_payload["image_assessment_display"] = image_assessment_display

    # Provide explicit raw section for debugging / logging
    ml_payload["raw_outputs"] = {"severity": severity_result, "disease": disease_result}

    # Triage rules
    triage = {
        "triage_level": "self_care",
        "red_flags": [],
        "needs_clinician_review": False,
    }
    quality = gate_result.get("quality", {})

    if clear_skin_suppressed:
        triage["suppressed_by_gate"] = True
    else:
        if quality.get("quality_status") == "poor":
            triage.update({"triage_level": "routine_review", "needs_clinician_review": True})
            triage["red_flags"].append("poor_image_quality")
            gemini_metrics.inc_priority_review()
        gate_top = gate_result.get("top_label")
        if gate_top == "rash_like_skin":
            triage["triage_level"] = "priority_review"
            triage["needs_clinician_review"] = True
            gemini_metrics.inc_priority_review()
        severity_cls = severity_result.get("predicted_class")
        if severity_cls == "severe" and not severity_result.get("severity_uncertain"):
            triage["triage_level"] = "priority_review"
            triage["needs_clinician_review"] = True
            gemini_metrics.inc_priority_review()
            if severity_result.get("confidence", 0) >= 0.8:
                triage["triage_level"] = "urgent_attention"
                triage["red_flags"].append("high_severity_confidence")
                triage["needs_clinician_review"] = True
                gemini_metrics.inc_urgent_attention()
                logger.warning("triage_escalation", extra={"triage": triage})
    ml_payload["triage"] = triage

    # Patient guidance
    guidance = {
        "summary": "Preliminary AI screening only. This is not a confirmed diagnosis.",
        "next_step": "Consider dermatologist review if symptoms persist.",
        "retake_required": quality.get("retake_required", False),
        "urgent_warning": triage["triage_level"] == "urgent_attention",
    }
    if quality.get("retake_required"):
        guidance["next_step"] = "Please retake a clear, well-lit close-up image of the area."
    if triage["triage_level"] in ("priority_review", "urgent_attention"):
        guidance["next_step"] = "Seek clinician review promptly."
    if clear_skin_suppressed:
        guidance.update(
            {
                "summary": "No obvious rash detected from image. This is not a confirmed diagnosis.",
                "next_step": "Monitor symptoms or retake a closer image only if a visible skin change is present.",
                "urgent_warning": False,
            }
        )
    ml_payload["patient_guidance"] = guidance

    response = {
        "success": True,
        "image_gate": gate_result,
        "ml_analysis": ml_payload,
        "message": "Analysis completed." if gate_status != "fail_open" else "Gate unavailable, ML analysis completed with fail-open.",
    }

    # Backward compatible: expose legacy keys alongside structured payload
    response.update(ml_payload)
    return response
