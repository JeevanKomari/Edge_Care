python -m venv venv
venv\Scripts\activate        # Windows

Step 2: Install Dependencies

pip install -r requirements.txt


Step 3: Prepare Dataset

Collect skin/rash images (JPEG/PNG)
Manually place images into folders:

dataset/train/mild/
dataset/train/moderate/
dataset/train/severe/

dataset/val/mild/
dataset/val/moderate/
dataset/val/severe/

Step 4: Validate Dataset (Optional but Recommended)
python training/prepare_data.py

This step:
Checks folder structure
Counts images per class
Verifies image readability

Step 5: Train CNN Model (ONE TIME)
python training/train_image_model.py

What happens internally:
Uses MobileNetV2 (transfer learning)
Trains classifier for mild / moderate / severe

Saves model as:

model/skin_model.h5
model/skin_model.tflite
This step may take a few minutes depending on system.

Step 6: Evaluate Model Performance
python training/evaluate.py

Step 7: Run FastAPI Application
uvicorn app.main:app --reload

http://127.0.0.1:8000/docs