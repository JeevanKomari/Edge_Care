#!/usr/bin/env python
"""
Threshold tuning helper for the HF image gate.
Example:
  python app/scripts/tune_hf_gate.py dataset/gate_labels.json \
    --threshold-set "{\"name\":\"tighter\",\"min_skin_score\":0.4,\"max_non_skin_score\":0.25}" \
    --output gate_tuning_results.json
"""

from app.services.hf_gate_tuning import main

if __name__ == "__main__":
    main()
