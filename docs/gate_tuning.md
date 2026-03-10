# HF Image Gate Threshold Tuning

## Dataset format
JSON or CSV with columns:
- `image_path` (required)
- `expected_gate_label` (`accepted`|`rejected`)
- `expected_reason` (optional)
- `split` (optional, e.g., train/val/test)
- `notes` (optional)

## Run the tuner
```bash
python app/scripts/tune_hf_gate.py dataset/gate_labels.json \
  --threshold-set "{\"name\":\"tighter\",\"min_skin_score\":0.4,\"max_non_skin_score\":0.25}" \
  --threshold-set "{\"name\":\"looser\",\"min_skin_score\":0.3,\"max_non_skin_score\":0.45}" \
  --output gate_tuning_results.json
```

## Interpreting results
- **false_accept**: gate allowed a sample that should be rejected (most critical for non-skin/screenshot safety).
- **false_reject**: gate blocked a valid skin image.
- Ranking sorts by lowest false_accept, then lowest false_reject, then highest balanced_accuracy.
- Start by lowering false_accept on non-skin/screenshot images, then reduce false_reject on valid skin.

## Tips
- Reuse cached images locally; the tuner calls the live HF zero-shot endpoint, so keep sample size moderate on free tiers.
- You can supply threshold sets via `--threshold-file path/to/sets.json` (list of objects with threshold overrides).
- Threshold keys match env vars (min_skin_score, max_non_skin_score, max_screenshot_score, max_blurry_score, min_rash_or_normal_score).
