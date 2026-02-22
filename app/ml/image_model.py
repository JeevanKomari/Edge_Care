import os
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf

MODEL_PATH = "model/skin_model.tflite"

interpreter = None
input_details = None
output_details = None

# Load model ONLY if it exists
if os.path.exists(MODEL_PATH):
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
else:
    print("⚠️ Model not found. Please train the model first.")


def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image.astype(np.float32)


def is_image_blurry(image, threshold=100.0):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold


CLASS_NAMES = ["mild", "moderate", "severe"]

def analyze_image(file):
    image = Image.open(file.file).convert("RGB")

    if is_image_blurry(image):
        return {
            "error": "Image quality too low. Please retake the photo."
        }

    if interpreter is None:
        return {
            "error": "Model not trained yet. Please run training/train_image_model.py"
        }

    # Preprocess
    input_data = preprocess_image(image)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]['index'])[0]

    # ✅ Model-driven prediction
    predicted_index = int(np.argmax(prediction))
    predicted_label = CLASS_NAMES[predicted_index]
    confidence = float(prediction[predicted_index])

    return {
        "predicted_class": predicted_label,
        "confidence": round(confidence, 2),
        "all_probabilities": {
            "mild": round(float(prediction[0]), 2),
            "moderate": round(float(prediction[1]), 2),
            "severe": round(float(prediction[2]), 2)
        }
    }