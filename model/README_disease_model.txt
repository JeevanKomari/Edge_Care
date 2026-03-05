EdgeCare Disease Name Model Files

Place the following files in this folder:

1) edgecare_disease_model.tflite
   - TFLite model trained on DermNet subset (10 classes)

2) edgecare_disease_labels.json
   - JSON list of class names in the SAME order as model outputs.

Optional:
3) edgecare_disease_config.json
   - {"threshold": 0.40, "label_map": {...}}
   - If missing, backend uses default threshold=0.40 and built-in label map.

After adding files, /ml/analyze-image will return both:
- severity: mild/moderate/severe (existing)
- disease: rash_detected/no_rash_detected + disease_name/confidence
