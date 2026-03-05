import os
import json
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf

tflite = tf.lite

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# ============================================================
# 1) Existing Severity Model (mild/moderate/severe)
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
        print("⚠️ Severity model not found at:", SEVERITY_MODEL_PATH)
except Exception as e:
    severity_interpreter = None
    severity_input_details = None
    severity_output_details = None
    print(f"⚠️ Severity model load failed: {e}")

# ============================================================
# 2) Disease Model (DermNet subset)
# ============================================================
DISEASE_MODEL_PATH = os.path.join(MODEL_DIR, "edgecare_disease_model.tflite")
DISEASE_LABELS_PATH = os.path.join(MODEL_DIR, "edgecare_disease_labels.json")
DISEASE_CONFIG_PATH = os.path.join(MODEL_DIR, "edgecare_disease_config.json")

DISEASE_THRESHOLD_DEFAULT = 0.28
DISEASE_TOP_K_DEFAULT = 3
MIN_SEVERITY_CONF_DEFAULT = 0.34

disease_interpreter = None
disease_input_details = None
disease_output_details = None

disease_class_names = None
disease_threshold = DISEASE_THRESHOLD_DEFAULT
disease_top_k = DISEASE_TOP_K_DEFAULT
min_severity_conf_for_rash = MIN_SEVERITY_CONF_DEFAULT

# Folder-name labels -> clean user-facing names
DEFAULT_LABEL_MAP = {
    "Acne and Rosacea Photos": "Acne / Rosacea",
    "Cellulitis Impetigo and other Bacterial Infections": "Bacterial infection (Impetigo/Cellulitis)",
    "Eczema Photos": "Eczema",
    "Light Diseases and Disorders of Pigmentation": "Pigmentation disorder (Vitiligo-like)",
    "Melanoma Skin Cancer Nevi and Moles": "Melanoma / Mole",
    "Psoriasis pictures Lichen Planus and related diseases": "Psoriasis / Lichen planus",
    "Seborrheic Keratoses and other Benign Tumors": "Benign tumor (Seborrheic keratosis)",
    "Tinea Ringworm Candidiasis and other Fungal Infections": "Fungal infection (Ringworm/Tinea)",
    "Urticaria Hives": "Hives (Urticaria)",
    "Warts Molluscum and other Viral Infections": "Viral infection (Warts/Molluscum)",
}

label_map = DEFAULT_LABEL_MAP.copy()


def _safe_load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _init_disease_model():
    global disease_interpreter, disease_input_details, disease_output_details
    global disease_class_names, disease_threshold, disease_top_k, min_severity_conf_for_rash, label_map

    if not os.path.exists(DISEASE_MODEL_PATH):
        print("⚠️ Disease model not found at:", DISEASE_MODEL_PATH)
        return

    try:
        disease_interpreter = tflite.Interpreter(model_path=DISEASE_MODEL_PATH)
        disease_interpreter.allocate_tensors()
        disease_input_details = disease_interpreter.get_input_details()
        disease_output_details = disease_interpreter.get_output_details()
    except Exception as e:
        disease_interpreter = None
        disease_input_details = None
        disease_output_details = None
        disease_class_names = None
        disease_threshold = DISEASE_THRESHOLD_DEFAULT
        disease_top_k = DISEASE_TOP_K_DEFAULT
        min_severity_conf_for_rash = MIN_SEVERITY_CONF_DEFAULT
        label_map = DEFAULT_LABEL_MAP.copy()
        print(f"⚠️ Disease model load failed: {e}")
        return

    # labels
    labels = _safe_load_json(DISEASE_LABELS_PATH)
    if isinstance(labels, list) and len(labels) > 0:
        disease_class_names = labels
    else:
        disease_class_names = None
        print("⚠️ Disease labels JSON not found/invalid at:", DISEASE_LABELS_PATH)

    # optional config (threshold, top_k, label map)
    cfg = _safe_load_json(DISEASE_CONFIG_PATH)
    if isinstance(cfg, dict):
        th = cfg.get("threshold")
        if isinstance(th, (int, float)):
            disease_threshold = float(th)

        tk = cfg.get("top_k")
        if isinstance(tk, int) and tk > 0:
            disease_top_k = tk

        ms = cfg.get("min_severity_conf_for_rash")
        if isinstance(ms, (int, float)):
            min_severity_conf_for_rash = float(ms)

        short_map = cfg.get("label_short_map")
        lm = cfg.get("label_map")
        if isinstance(short_map, dict) and short_map:
            label_map = {**label_map, **short_map}
        elif isinstance(lm, dict) and lm:
            label_map = {**label_map, **lm}

        # fall back to config classes if labels missing
        classes_cfg = cfg.get("classes")
        if disease_class_names is None and isinstance(classes_cfg, list) and classes_cfg:
            disease_class_names = classes_cfg


_init_disease_model()


# ============================================================
# Common helpers
# ============================================================
def preprocess_image_224(image: Image.Image) -> np.ndarray:
    """Resize to 224x224, normalize to [0,1], add batch dim."""
    image = image.resize((224, 224))
    arr = np.array(image) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr.astype(np.float32)


def is_image_blurry(image: Image.Image, threshold: float = 100.0) -> bool:
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold


def _softmax_if_needed(x: np.ndarray) -> np.ndarray:
    """Some TFLite exports already output softmax; this keeps it safe."""
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


def _get_top_predictions(probabilities: np.ndarray, top_k: int):
    probs = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    sorted_idx = np.argsort(probs)[::-1][:top_k]
    preds = []
    for idx in sorted_idx:
        score = float(probs[idx])
        raw_label = (
            disease_class_names[idx]
            if disease_class_names and 0 <= idx < len(disease_class_names)
            else f"class_{idx}"
        )
        friendly = label_map.get(raw_label, raw_label)
        preds.append({"label": raw_label, "name": friendly, "score": round(score, 4)})
    return preds


def _predict_disease(input_data: np.ndarray, severity_max_prob: float):
    if disease_interpreter is None:
        return {
            "status": "disabled",
            "message": "Disease model not loaded on server.",
        }

    try:
        disease_interpreter.set_tensor(disease_input_details[0]["index"], input_data)
        disease_interpreter.invoke()
        raw = disease_interpreter.get_tensor(disease_output_details[0]["index"])[0]
    except Exception as e:
        return {
            "status": "disabled",
            "message": f"Disease model inference failed: {e}",
        }

    probs = _softmax_if_needed(raw)
    top_predictions = _get_top_predictions(probs, disease_top_k)

    if not top_predictions:
        return {
            "status": "disabled",
            "message": "Disease model returned no predictions.",
        }

    top1 = top_predictions[0]
    top1_score = float(top1["score"])
    best_guess = top1["name"]
    raw_label = top1["label"]

    if top1_score >= disease_threshold:
        return {
            "status": "rash_detected",
            "disease_name": best_guess,
            "confidence": round(top1_score, 4),
            "raw_label": raw_label,
            "top_predictions": top_predictions,
        }

    if severity_max_prob >= min_severity_conf_for_rash:
        return {
            "status": "uncertain",
            "message": "Rash likely detected but disease type is uncertain. Upload a closer, well-lit photo focused on the affected area.",
            "best_guess": best_guess,
            "confidence": round(top1_score, 4),
            "threshold": round(disease_threshold, 4),
            "top_predictions": top_predictions,
        }

    return {
        "status": "no_rash_detected",
        "message": "No rash detected or image is unclear. Please upload a clear rash/skin issue photo.",
        "best_guess": best_guess,
        "confidence": round(top1_score, 4),
        "threshold": round(disease_threshold, 4),
        "top_predictions": top_predictions,
    }


# ============================================================
# Public API called by /ml/analyze-image
# ============================================================
def analyze_image(file):
    image = Image.open(file.file).convert("RGB")

    if is_image_blurry(image):
        return {"error": "Image quality too low. Please retake the photo."}

    input_data = preprocess_image_224(image)

    severity_result, severity_err = _predict_severity(input_data)
    if severity_err:
        return severity_err

    severity_probs = severity_result.get("all_probabilities", {})
    severity_max_prob = max(severity_probs.values()) if severity_probs else 0.0

    disease_result = _predict_disease(input_data, severity_max_prob)

    # Backward compatible response + new fields
    resp = dict(severity_result)
    resp["disease"] = disease_result
    return resp
