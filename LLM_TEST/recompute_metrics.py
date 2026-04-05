import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from config_utils import get_section, load_env_file, load_yaml_config, resolve_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge LLM re-check results into the original predictions and recompute metrics."
    )
    parser.add_argument("--config", default="LLM_TEST/exp.yaml", help="Path to YAML config.")
    parser.add_argument("--env_file", default="LLM_TEST/.env", help="Path to environment config file.")
    parser.add_argument("--results_csv", default=None, help="Path to the original results.csv.")
    parser.add_argument(
        "--llm_predictions_csv",
        default=None,
        help="Path to LLM_TEST/output/<dataset_id>/llm_predictions.csv",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory for merged outputs. Defaults to the parent directory of --llm_predictions_csv.",
    )
    return parser.parse_args()


def load_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_int(value: object, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def compute_metrics(labels: List[int], preds: List[int]) -> Dict[str, float]:
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)

    total = len(labels)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total": total,
    }


def safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    config = load_yaml_config(args.config)
    metrics_cfg = get_section(config, "metrics")

    results_csv = Path(resolve_value(args.results_csv, metrics_cfg, "results_csv")).resolve()
    llm_predictions_csv = Path(
        resolve_value(args.llm_predictions_csv, metrics_cfg, "llm_predictions_csv")
    ).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else Path(resolve_value(None, metrics_cfg, "output_dir", llm_predictions_csv.parent)).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    original_rows = load_csv(results_csv)
    llm_rows = load_csv(llm_predictions_csv)
    llm_by_index = {to_int(row.get("Index")): row for row in llm_rows}

    merged_rows: List[dict] = []
    positive_case_rows: List[dict] = []
    original_labels: List[int] = []
    original_preds: List[int] = []
    merged_preds: List[int] = []
    reviewed_labels: List[int] = []
    reviewed_original_preds: List[int] = []
    reviewed_final_preds: List[int] = []

    llm_override_count = 0
    llm_missing_count = 0
    actual_positive_total = 0
    actual_positive_reviewed_by_llm = 0
    actual_positive_missed_by_original = 0
    actual_positive_kept_by_llm = 0
    actual_positive_rejected_by_llm = 0

    for row in original_rows:
        index = to_int(row.get("Index"))
        label = to_int(row.get("Label"))
        original_pred = to_int(row.get("Prediction"))

        final_pred = original_pred
        llm_row = llm_by_index.get(index)

        if original_pred == 1:
            reviewed_labels.append(label)
            reviewed_original_preds.append(original_pred)
            if llm_row and str(llm_row.get("LLMPrediction", "")).strip() in {"0", "1"}:
                final_pred = to_int(llm_row.get("LLMPrediction"))
                llm_override_count += 1
            else:
                llm_missing_count += 1
            reviewed_final_preds.append(final_pred)

        original_labels.append(label)
        original_preds.append(original_pred)
        merged_preds.append(final_pred)

        merged_rows.append(
            {
                "Index": index,
                "CWE": row.get("CWE", "null"),
                "Label": label,
                "OriginalPrediction": original_pred,
                "FinalPrediction": final_pred,
                "OriginalProb": row.get("Prob", ""),
                "LLMPrediction": "" if not llm_row else llm_row.get("LLMPrediction", ""),
                "LLMModel": "" if not llm_row else llm_row.get("LLMModel", ""),
                "LLMParseStatus": "" if not llm_row else llm_row.get("ParseStatus", ""),
            }
        )

        if label == 1:
            actual_positive_total += 1
            llm_prediction = "" if not llm_row else llm_row.get("LLMPrediction", "")
            parse_status = "" if not llm_row else llm_row.get("ParseStatus", "")
            llm_model = "" if not llm_row else llm_row.get("LLMModel", "")

            if original_pred == 1:
                actual_positive_reviewed_by_llm += 1
                if final_pred == 1:
                    positive_case_status = "true_positive_kept_by_llm"
                    actual_positive_kept_by_llm += 1
                else:
                    positive_case_status = "true_positive_rejected_by_llm"
                    actual_positive_rejected_by_llm += 1
            else:
                positive_case_status = "missed_by_original_model"
                actual_positive_missed_by_original += 1

            positive_case_rows.append(
                {
                    "Index": index,
                    "CWE": row.get("CWE", "null"),
                    "Label": label,
                    "OriginalPrediction": original_pred,
                    "FinalPrediction": final_pred,
                    "OriginalProb": row.get("Prob", ""),
                    "LLMPrediction": llm_prediction,
                    "LLMModel": llm_model,
                    "LLMParseStatus": parse_status,
                    "PositiveCaseStatus": positive_case_status,
                }
            )

    original_metrics = compute_metrics(original_labels, original_preds)
    merged_metrics = compute_metrics(original_labels, merged_preds)
    reviewed_subset_original_metrics = compute_metrics(reviewed_labels, reviewed_original_preds)
    reviewed_subset_final_metrics = compute_metrics(reviewed_labels, reviewed_final_preds)
    positive_case_summary = {
        "actual_positive_total": actual_positive_total,
        "actual_positive_reviewed_by_llm": actual_positive_reviewed_by_llm,
        "actual_positive_missed_by_original": actual_positive_missed_by_original,
        "actual_positive_kept_by_llm": actual_positive_kept_by_llm,
        "actual_positive_rejected_by_llm": actual_positive_rejected_by_llm,
        "original_recall_on_actual_positives": safe_rate(
            actual_positive_reviewed_by_llm, actual_positive_total
        ),
        "final_recall_on_actual_positives": safe_rate(
            actual_positive_kept_by_llm, actual_positive_total
        ),
        "llm_keep_rate_within_reviewed_actual_positives": safe_rate(
            actual_positive_kept_by_llm, actual_positive_reviewed_by_llm
        ),
        "llm_reject_rate_within_reviewed_actual_positives": safe_rate(
            actual_positive_rejected_by_llm, actual_positive_reviewed_by_llm
        ),
    }

    merged_csv_path = output_dir / "merged_results.csv"
    with merged_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Index",
                "CWE",
                "Label",
                "OriginalPrediction",
                "FinalPrediction",
                "OriginalProb",
                "LLMPrediction",
                "LLMModel",
                "LLMParseStatus",
            ],
        )
        writer.writeheader()
        writer.writerows(merged_rows)

    positive_cases_path = output_dir / "positive_case_details.csv"
    with positive_cases_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Index",
                "CWE",
                "Label",
                "OriginalPrediction",
                "FinalPrediction",
                "OriginalProb",
                "LLMPrediction",
                "LLMModel",
                "LLMParseStatus",
                "PositiveCaseStatus",
            ],
        )
        writer.writeheader()
        writer.writerows(positive_case_rows)

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "results_csv": str(results_csv),
                "llm_predictions_csv": str(llm_predictions_csv),
                "metrics_scope": {
                    "original_metrics": "computed on all test cases",
                    "final_metrics": "computed on all test cases",
                    "reviewed_subset_original_metrics": "computed only on samples with OriginalPrediction == 1",
                    "reviewed_subset_final_metrics": "computed only on samples with OriginalPrediction == 1",
                },
                "llm_override_count": llm_override_count,
                "llm_missing_count": llm_missing_count,
                "original_metrics": original_metrics,
                "final_metrics": merged_metrics,
                "reviewed_subset_original_metrics": reviewed_subset_original_metrics,
                "reviewed_subset_final_metrics": reviewed_subset_final_metrics,
                "positive_case_summary": positive_case_summary,
                "positive_case_details_csv": str(positive_cases_path),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print("original_metrics")
    print(json.dumps(original_metrics, ensure_ascii=False, indent=2))
    print("final_metrics")
    print(json.dumps(merged_metrics, ensure_ascii=False, indent=2))
    print("reviewed_subset_original_metrics")
    print(json.dumps(reviewed_subset_original_metrics, ensure_ascii=False, indent=2))
    print("reviewed_subset_final_metrics")
    print(json.dumps(reviewed_subset_final_metrics, ensure_ascii=False, indent=2))
    print("positive_case_summary")
    print(json.dumps(positive_case_summary, ensure_ascii=False, indent=2))
    print(f"merged_results={merged_csv_path}")
    print(f"positive_case_details={positive_cases_path}")
    print(f"metrics_json={metrics_path}")


if __name__ == "__main__":
    main()
