import os
import json
import numpy as np
from PIL import Image
import cv2
import tflite_runtime.interpreter as tflite

# ============================================================
# 1) Existing Severity Model (mild/moderate/severe)
# ============================================================
SEVERITY_MODEL_PATH = "model/skin_model.tflite"
SEVERITY_CLASS_NAMES = ["mild", "moderate", "severe"]

severity_interpreter = None
severity_input_details = None
severity_output_details = None

if os.path.exists(SEVERITY_MODEL_PATH):
    severity_interpreter = tflite.Interpreter(model_path=SEVERITY_MODEL_PATH)
    severity_interpreter.allocate_tensors()
    severity_input_details = severity_interpreter.get_input_details()
    severity_output_details = severity_interpreter.get_output_details()
else:
    print("⚠️ Severity model not found at:", SEVERITY_MODEL_PATH)

# ============================================================
# 2) New Disease Name Model (DermNet subset)
# ============================================================
DISEASE_MODEL_PATH = "model/edgecare_disease_model.tflite"
DISEASE_LABELS_PATH = "model/edgecare_disease_labels.json"
DISEASE_CONFIG_PATH = "model/edgecare_disease_config.json"

DISEASE_THRESHOLD_DEFAULT = 0.40  # 'No rash / unclear image' gate

disease_interpreter = None
disease_input_details = None
disease_output_details = None

disease_class_names = None
disease_threshold = DISEASE_THRESHOLD_DEFAULT

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
    global disease_class_names, disease_threshold, label_map

    if not os.path.exists(DISEASE_MODEL_PATH):
        print("⚠️ Disease model not found at:", DISEASE_MODEL_PATH)
        return

    disease_interpreter = tflite.Interpreter(model_path=DISEASE_MODEL_PATH)
    disease_interpreter.allocate_tensors()
    disease_input_details = disease_interpreter.get_input_details()
    disease_output_details = disease_interpreter.get_output_details()

    # labels
    labels = _safe_load_json(DISEASE_LABELS_PATH)
    if isinstance(labels, list) and len(labels) > 0:
        disease_class_names = labels
    else:
        disease_class_names = None
        print("⚠️ Disease labels JSON not found/invalid at:", DISEASE_LABELS_PATH)

    # optional config (threshold + label map)
    cfg = _safe_load_json(DISEASE_CONFIG_PATH)
    if isinstance(cfg, dict):
        th = cfg.get("threshold")
        if isinstance(th, (int, float)):
            disease_threshold = float(th)
        lm = cfg.get("label_map")
        if isinstance(lm, dict) and lm:
            label_map = {**label_map, **lm}

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
    # if sums to ~1, assume it's already probabilities
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
    if disease_interpreter is None:
        return None  # disease model optional

    disease_interpreter.set_tensor(disease_input_details[0]["index"], input_data)
    disease_interpreter.invoke()
    raw = disease_interpreter.get_tensor(disease_output_details[0]["index"])[0]
    probs = _softmax_if_needed(raw)

    idx = int(np.argmax(probs))
    conf = float(probs[idx])

    raw_label = None
    if disease_class_names and 0 <= idx < len(disease_class_names):
        raw_label = disease_class_names[idx]
    else:
        raw_label = f"class_{idx}"

    clean_label = label_map.get(raw_label, raw_label)

    # 'No rash / unclear' gate
    if conf < disease_threshold:
        return {
            "status": "no_rash_detected",
            "message": "No rash detected or image is unclear. Please upload a clear rash/skin issue photo.",
            "confidence": round(conf, 4),
            "best_guess": clean_label,
            "threshold": round(disease_threshold, 4),
        }

    return {
        "status": "rash_detected",
        "disease_name": clean_label,
        "confidence": round(conf, 4),
        "raw_label": raw_label,
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

    disease_result = _predict_disease(input_data)

    # Backward compatible response + new fields
    resp = dict(severity_result)
    resp["disease"] = disease_result  # can be None if model not present
    return resp
