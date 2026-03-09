
import json
import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_PATH = "edgecare_6class_effnetb0_best.keras"
CLASS_NAMES_PATH = "class_names.json"
IMG_SIZE = (224, 224)

def load_artifacts(model_path=MODEL_PATH, class_names_path=CLASS_NAMES_PATH):
    model = tf.keras.models.load_model(model_path)
    with open(class_names_path, "r") as f:
        class_names = json.load(f)
    return model, class_names

def preprocess_image(image_path, img_size=IMG_SIZE):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(img_size)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_image(model, class_names, image_path):
    x = preprocess_image(image_path)
    probs = model.predict(x, verbose=0)[0]
    top_idx = int(np.argmax(probs))
    result = {
        "predicted_class": class_names[top_idx],
        "confidence": float(probs[top_idx]),
        "all_probabilities": {
            class_names[i]: float(probs[i]) for i in range(len(class_names))
        }
    }
    return result
