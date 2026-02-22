import os
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

CLASSES = ["mild", "moderate", "severe"]
SPLITS = ["train", "val"]

IMG_SIZE = (224, 224)

def prepare_dataset():
    print("🔍 Checking dataset structure...\n")

    for split in SPLITS:
        for cls in CLASSES:
            folder = os.path.join(DATASET_DIR, split, cls)
            if not os.path.exists(folder):
                raise Exception(f"❌ Missing folder: {folder}")

            images = os.listdir(folder)
            print(f"{split}/{cls}: {len(images)} images")

            for img_name in images:
                img_path = os.path.join(folder, img_name)
                try:
                    img = Image.open(img_path).convert("RGB")
                    img.resize(IMG_SIZE)
                except Exception:
                    print(f"⚠️ Problem with image: {img_path}")

    print("\n✅ Dataset structure validated successfully.")

if __name__ == "__main__":
    prepare_dataset()
