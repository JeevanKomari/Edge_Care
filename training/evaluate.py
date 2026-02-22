import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "skin_model.h5")
VAL_DIR = os.path.join(BASE_DIR, "dataset", "val")

IMG_SIZE = 224
BATCH_SIZE = 16

def evaluate_model():
    if not os.path.exists(MODEL_PATH):
        raise Exception("❌ Model not found. Train model first.")

    model = tf.keras.models.load_model(MODEL_PATH)

    datagen = ImageDataGenerator(rescale=1.0 / 255)

    val_data = datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    predictions = model.predict(val_data)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_data.classes

    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=val_data.class_indices.keys()))

    print("\n📉 Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    evaluate_model()
