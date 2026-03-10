import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

from app.services.hf_image_gate import GateThresholds, evaluate_gate_scores, run_image_gate


@dataclass
class ThresholdSet:
    name: str
    thresholds: GateThresholds


@dataclass
class SampleRow:
    image_path: str
    expected_gate_label: str
    expected_reason: str | None
    split: str | None
    notes: str | None


def _load_rows(path: str) -> List[SampleRow]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    rows: List[SampleRow] = []
    if path.lower().endswith(".json"):
        data = json.load(open(path, "r", encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("JSON dataset must be a list of rows")
        for item in data:
            rows.append(
                SampleRow(
                    image_path=item.get("image_path"),
                    expected_gate_label=str(item.get("expected_gate_label", "")).lower(),
                    expected_reason=item.get("expected_reason"),
                    split=item.get("split"),
                    notes=item.get("notes"),
                )
            )
    else:  # csv
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for item in reader:
                rows.append(
                    SampleRow(
                        image_path=item.get("image_path"),
                        expected_gate_label=str(item.get("expected_gate_label", "")).lower(),
                        expected_reason=item.get("expected_reason"),
                        split=item.get("split"),
                        notes=item.get("notes"),
                    )
                )
    return rows


def _merge_thresholds(base: GateThresholds, overrides: Dict[str, Any]) -> GateThresholds:
    data = base.__dict__.copy()
    for key, val in overrides.items():
        if key in data:
            try:
                data[key] = float(val)
            except (TypeError, ValueError):
                pass
    return GateThresholds(**data)


def parse_threshold_sets(raw_sets: List[Dict[str, Any]]) -> List[ThresholdSet]:
    base = GateThresholds()
    sets: List[ThresholdSet] = []
    for idx, raw in enumerate(raw_sets):
        name = raw.get("name") or f"set_{idx+1}"
        thresholds = _merge_thresholds(base, raw)
        sets.append(ThresholdSet(name=name, thresholds=thresholds))
    if not sets:
        sets.append(ThresholdSet(name="default", thresholds=base))
    return sets


def _infer_scores(image_path: str) -> Tuple[List[Dict[str, Any]], str]:
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    gate_result = run_image_gate(image_bytes, filename=os.path.basename(image_path), record_metrics=False)
    status = gate_result.get("status")
    return gate_result.get("scores", []), status


def evaluate_threshold_sets(
    dataset_path: str,
    threshold_sets: List[ThresholdSet],
    infer_scores_fn: Callable[[str], Tuple[List[Dict[str, Any]], str]] | None = None,
) -> Dict[str, Any]:
    infer_scores_fn = infer_scores_fn or _infer_scores
    rows = _load_rows(dataset_path)
    results: List[Dict[str, Any]] = []

    for tset in threshold_sets:
        results.append(
            {
                "name": tset.name,
                "thresholds": tset.thresholds.__dict__,
                "counts": {
                    "true_accept": 0,
                    "true_reject": 0,
                    "false_accept": 0,
                    "false_reject": 0,
                    "errors": 0,
                },
                "reason_counts": {},
                "top_labels": {},
            }
        )

    for row in rows:
        try:
            scores, base_status = infer_scores_fn(row.image_path)
        except Exception:
            for r in results:
                r["counts"]["errors"] += 1
            continue

        if base_status in {"error", "fail_open"}:
            for r in results:
                r["counts"]["errors"] += 1
            continue

        expected_accept = row.expected_gate_label == "accepted"

        for r, tset in zip(results, threshold_sets):
            decision = evaluate_gate_scores(scores, tset.thresholds)
            pred_accept = decision.accepted
            if pred_accept and expected_accept:
                r["counts"]["true_accept"] += 1
            elif (not pred_accept) and (not expected_accept):
                r["counts"]["true_reject"] += 1
            elif pred_accept and (not expected_accept):
                r["counts"]["false_accept"] += 1
            else:
                r["counts"]["false_reject"] += 1

            for code in decision.reason_codes or ["NONE"]:
                r["reason_counts"][code] = r["reason_counts"].get(code, 0) + 1
            if decision.top_label:
                r["top_labels"][decision.top_label] = r["top_labels"].get(decision.top_label, 0) + 1

    for r in results:
        counts = r["counts"]
        total = sum(counts.values()) - counts["errors"]
        positives = counts["true_accept"] + counts["false_reject"]
        negatives = counts["true_reject"] + counts["false_accept"]
        precision_accept = counts["true_accept"] / max(counts["true_accept"] + counts["false_accept"], 1)
        recall_accept = counts["true_accept"] / max(positives, 1)
        precision_reject = counts["true_reject"] / max(counts["true_reject"] + counts["false_reject"], 1)
        recall_reject = counts["true_reject"] / max(negatives, 1)
        accuracy = (counts["true_accept"] + counts["true_reject"]) / max(total, 1)
        balanced_accuracy = (recall_accept + recall_reject) / 2
        r["metrics"] = {
            "accuracy": round(accuracy, 4),
            "precision_accept": round(precision_accept, 4),
            "recall_accept": round(recall_accept, 4),
            "precision_reject": round(precision_reject, 4),
            "recall_reject": round(recall_reject, 4),
            "balanced_accuracy": round(balanced_accuracy, 4),
        }

    ranked = sorted(
        results,
        key=lambda x: (
            x["counts"]["false_accept"],
            x["counts"]["false_reject"],
            -x["metrics"]["balanced_accuracy"],
        ),
    )

    return {"ranked_results": ranked, "total_samples": len(rows)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate HF image gate thresholds")
    parser.add_argument("dataset", help="Path to JSON or CSV with labeled rows")
    parser.add_argument("--threshold-set", action="append", help="Threshold overrides as JSON string")
    parser.add_argument("--threshold-file", help="JSON file containing list of threshold sets")
    parser.add_argument("--output", default="gate_tuning_results.json", help="Where to save results JSON")
    args = parser.parse_args()

    raw_sets: List[Dict[str, Any]] = []
    if args.threshold_set:
        for item in args.threshold_set:
            raw_sets.append(json.loads(item))
    if args.threshold_file:
        raw_sets.extend(json.load(open(args.threshold_file, "r", encoding="utf-8")))

    threshold_sets = parse_threshold_sets(raw_sets)
    summary = evaluate_threshold_sets(args.dataset, threshold_sets)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
