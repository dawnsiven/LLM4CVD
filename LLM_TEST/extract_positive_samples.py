import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

from config_utils import get_section, load_env_file, load_yaml_config, resolve_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract positive predictions from results.csv and map them to the source test JSON."
    )
    parser.add_argument("--config", default="LLM_TEST/exp.yaml", help="Path to YAML config.")
    parser.add_argument("--env_file", default="LLM_TEST/.env", help="Path to environment config file.")
    parser.add_argument("--results_csv", default=None, help="Path to the original results.csv file.")
    parser.add_argument(
        "--data_json",
        default=None,
        help="Path to the source *_test.json file. If omitted, the script will try to infer it.",
    )
    parser.add_argument(
        "--data_root",
        default="data",
        help="Root directory used to infer the source *_test.json file when --data_json is omitted.",
    )
    parser.add_argument(
        "--output_root",
        default="LLM_TEST/intermediate",
        help="Root directory for intermediate outputs.",
    )
    return parser.parse_args()


def infer_dataset_id(results_csv: Path) -> str:
    return results_csv.parent.name


def infer_data_json(results_csv: Path, data_root: Path) -> Path:
    dataset_id = infer_dataset_id(results_csv)
    candidates = sorted(data_root.rglob(f"{dataset_id}_test.json"))
    if not candidates:
        raise FileNotFoundError(
            f"Could not infer data json for dataset_id={dataset_id} under {data_root}"
        )
    if len(candidates) > 1:
        raise FileExistsError(
            f"Found multiple candidate test json files for dataset_id={dataset_id}: "
            + ", ".join(str(path) for path in candidates)
        )
    return candidates[0]


def load_results(results_csv: Path) -> List[dict]:
    with results_csv.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(data_json: Path) -> List[dict]:
    with data_json.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_index_map(records: List[dict]) -> Dict[int, dict]:
    mapping: Dict[int, dict] = {}
    for record in records:
        record_index = record.get("index")
        if record_index is None:
            continue
        mapping[int(record_index)] = record
    return mapping


def to_int(value: object, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def safe_write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    config = load_yaml_config(args.config)
    common_cfg = get_section(config, "common")
    extract_cfg = get_section(config, "extract")

    results_csv = Path(resolve_value(args.results_csv, extract_cfg, "results_csv")).resolve()
    data_root = Path(resolve_value(args.data_root, common_cfg, "data_root", "data")).resolve()
    output_root = resolve_value(args.output_root, common_cfg, "intermediate_root", "LLM_TEST/intermediate")
    data_json_value = resolve_value(args.data_json, extract_cfg, "data_json")
    data_json = Path(data_json_value).resolve() if data_json_value else infer_data_json(results_csv, data_root)

    dataset_id = infer_dataset_id(results_csv)
    output_dir = Path(output_root).resolve() / dataset_id
    output_dir.mkdir(parents=True, exist_ok=True)

    result_rows = load_results(results_csv)
    data_rows = load_json(data_json)
    data_by_index = build_index_map(data_rows)

    positive_rows = [row for row in result_rows if to_int(row.get("Prediction")) == 1]

    matched_samples: List[dict] = []
    unmatched_indices: List[int] = []
    positive_indices: List[int] = []

    for row in positive_rows:
        sample_index = to_int(row.get("Index"))
        positive_indices.append(sample_index)
        sample = data_by_index.get(sample_index)
        if sample is None:
            unmatched_indices.append(sample_index)
            continue

        matched_samples.append(
            {
                "dataset_id": dataset_id,
                "index": sample_index,
                "cwe": row.get("CWE", sample.get("cwe", "null")),
                "ground_truth": to_int(row.get("Label"), default=to_int(sample.get("output"), default=0)),
                "original_prediction": to_int(row.get("Prediction")),
                "original_probability": row.get("Prob"),
                "instruction": sample.get("instruction", ""),
                "input": sample.get("input", ""),
                "output": sample.get("output", ""),
                "raw_sample": sample,
            }
        )

    safe_write_json(
        output_dir / "summary.json",
        {
            "dataset_id": dataset_id,
            "results_csv": str(results_csv),
            "data_json": str(data_json),
            "total_results": len(result_rows),
            "positive_predictions": len(positive_rows),
            "matched_positive_samples": len(matched_samples),
            "unmatched_positive_indices": len(unmatched_indices),
        },
    )
    safe_write_json(output_dir / "positive_indices.json", positive_indices)
    safe_write_json(output_dir / "unmatched_indices.json", unmatched_indices)
    safe_write_json(output_dir / "positive_samples.json", matched_samples)

    print(f"dataset_id={dataset_id}")
    print(f"results_csv={results_csv}")
    print(f"data_json={data_json}")
    print(f"positive_predictions={len(positive_rows)}")
    print(f"matched_positive_samples={len(matched_samples)}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
