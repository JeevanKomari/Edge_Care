import os
import json
import numpy as np
from PIL import Image, UnidentifiedImageError
import cv2
import tensorflow as tf

tflite = tf.lite

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_DIR = os.path.join(BASE_DIR, "model")
EXPORTS_DIR = os.path.join(BASE_DIR, "exports", "edgecare_6class_v1")

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

    return result


# ============================================================
# Public API called by /ml/analyze-image
# ============================================================
def analyze_image(file):
    try:
        image = Image.open(file.file).convert("RGB")
    except UnidentifiedImageError:
        return {"error": "Uploaded file is not a valid image."}
    except Exception as e:
        return {"error": f"Failed to read image: {e}"}

    if is_image_blurry(image):
        return {"error": "Image quality too low. Please retake the photo."}

    input_data = preprocess_image_224(image)

    severity_result, severity_err = _predict_severity(input_data)
    if severity_err:
        return severity_err

    disease_result = _predict_disease(input_data)

    # Backward compatible response: keep severity at top-level, nest disease result
    resp = dict(severity_result)
    resp["disease"] = disease_result
    return resp
