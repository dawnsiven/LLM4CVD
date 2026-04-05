import argparse
import csv
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config_utils import get_section, load_env_file, load_yaml_config, resolve_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call an OpenAI-compatible API to re-check extracted positive samples."
    )
    parser.add_argument("--config", default="LLM_TEST/exp.yaml", help="Path to YAML config.")
    parser.add_argument("--env_file", default="LLM_TEST/.env", help="Path to environment config file.")
    parser.add_argument(
        "--input_json",
        default=None,
        help="Path to LLM_TEST/intermediate/<dataset_id>/positive_samples.json",
    )
    parser.add_argument(
        "--prompt_file",
        default=None,
        help="Prompt template file. The code snippet will be appended automatically.",
    )
    parser.add_argument("--model", default=None, help="API model name.")
    parser.add_argument(
        "--api_base",
        default=None,
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--api_key",
        default=None,
        help="API key. Defaults to OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--output_root",
        default="LLM_TEST/output",
        help="Root directory for LLM outputs.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep_seconds", type=float, default=1.0)
    return parser.parse_args()


def load_json(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        return handle.read().strip()


def build_user_prompt(prompt_template: str, code: str) -> str:
    return (
        f"{prompt_template}\n\n"
        "Code:\n"
        "```c\n"
        f"{code}\n"
        "```\n"
    )


def extract_text_from_response(payload: dict) -> str:
    choices = payload.get("choices", [])
    if not choices:
        return ""

    first = choices[0]
    message = first.get("message", {})
    if isinstance(message, dict) and message.get("content"):
        return str(message["content"]).strip()

    text = first.get("text")
    return str(text).strip() if text is not None else ""


def try_parse_json_block(text: str) -> Optional[dict]:
    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates = [fenced.group(1)] if fenced else []

    braces = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if braces:
        candidates.append(braces.group(1))

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def parse_binary_label(text: str) -> Tuple[Optional[int], Optional[str]]:
    parsed = try_parse_json_block(text)
    if isinstance(parsed, dict):
        vulnerable = str(parsed.get("vulnerable", "")).strip().lower()
        if vulnerable in {"yes", "true", "1"}:
            return 1, "json.vulnerable"
        if vulnerable in {"no", "false", "0"}:
            return 0, "json.vulnerable"

    lowered = text.lower()
    if re.search(r'"vulnerable"\s*:\s*"?(yes|true|1)"?', lowered):
        return 1, "regex.vulnerable"
    if re.search(r'"vulnerable"\s*:\s*"?(no|false|0)"?', lowered):
        return 0, "regex.vulnerable"
    if re.search(r"\byes\b", lowered):
        return 1, "regex.yes"
    if re.search(r"\bno\b", lowered):
        return 0, "regex.no"
    return None, None


def call_chat_completion(
    api_base: str,
    api_key: str,
    model: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> dict:
    url = api_base.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "system",
                "content": "You are a careful vulnerability detection assistant. Return valid JSON whenever possible.",
            },
            {"role": "user", "content": user_prompt},
        ],
    }

    request = urllib.request.Request(
        url=url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[dict]) -> None:
    fieldnames = [
        "Index",
        "Label",
        "OriginalPrediction",
        "LLMPrediction",
        "LLMModel",
        "ParseStatus",
        "RawResponse",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    config = load_yaml_config(args.config)
    common_cfg = get_section(config, "common")
    llm_cfg = get_section(config, "llm")

    input_json_value = resolve_value(args.input_json, llm_cfg, "input_json")
    prompt_file_value = resolve_value(args.prompt_file, llm_cfg, "prompt_file")
    model = resolve_value(args.model, llm_cfg, "model")
    api_base = resolve_value(
        args.api_base,
        llm_cfg,
        "api_base",
        os.environ.get("OPENAI_API_BASE", "https://api.deepseek.com/v1"),
    )
    api_key_env = llm_cfg.get("api_key_env", "OPENAI_API_KEY")
    api_key = resolve_value(args.api_key, llm_cfg, "api_key", os.environ.get(api_key_env))
    output_root = resolve_value(args.output_root, common_cfg, "output_root", "LLM_TEST/output")
    temperature = float(resolve_value(args.temperature, llm_cfg, "temperature", 0.0))
    max_tokens = int(resolve_value(args.max_tokens, llm_cfg, "max_tokens", 256))
    timeout = int(resolve_value(args.timeout, llm_cfg, "timeout", 120))
    retries = int(resolve_value(args.retries, llm_cfg, "retries", 3))
    sleep_seconds = float(resolve_value(args.sleep_seconds, llm_cfg, "sleep_seconds", 1.0))

    if not api_key:
        raise ValueError(
            f"API key is required. Please set '{api_key_env}' in {Path(args.env_file).resolve()} "
            "or pass --api_key."
        )

    input_json = Path(input_json_value).resolve()
    prompt_file = Path(prompt_file_value).resolve()
    positive_samples = load_json(input_json)
    prompt_template = ensure_text(prompt_file)

    dataset_id = input_json.parent.name
    output_dir = Path(output_root).resolve() / dataset_id
    output_dir.mkdir(parents=True, exist_ok=True)

    judgment_rows: List[dict] = []
    csv_rows: List[dict] = []

    for sample in positive_samples:
        user_prompt = build_user_prompt(prompt_template, sample.get("input", ""))

        response_text = ""
        parse_status = "unparsed"
        llm_prediction: Optional[int] = None
        last_error = None

        for attempt in range(1, args.retries + 1):
            try:
                payload = call_chat_completion(
                    api_base=api_base,
                    api_key=api_key,
                    model=model,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
                response_text = extract_text_from_response(payload)
                llm_prediction, parse_source = parse_binary_label(response_text)
                parse_status = parse_source or "unparsed"
                if llm_prediction is None:
                    raise ValueError("Could not parse vulnerable=yes/no from model response.")
                break
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError) as exc:
                last_error = str(exc)
                if attempt == retries:
                    response_text = response_text or last_error or ""
                    parse_status = f"error:{type(exc).__name__}"
                    llm_prediction = None
                else:
                    time.sleep(sleep_seconds)

        judgment = {
            "dataset_id": dataset_id,
            "index": sample["index"],
            "label": sample["ground_truth"],
            "original_prediction": sample["original_prediction"],
            "llm_prediction": llm_prediction,
            "model": model,
            "parse_status": parse_status,
            "prompt_file": str(prompt_file),
            "raw_response": response_text,
        }
        judgment_rows.append(judgment)
        csv_rows.append(
            {
                "Index": sample["index"],
                "Label": sample["ground_truth"],
                "OriginalPrediction": sample["original_prediction"],
                "LLMPrediction": "" if llm_prediction is None else llm_prediction,
                "LLMModel": model,
                "ParseStatus": parse_status,
                "RawResponse": response_text,
            }
        )

        print(
            f"index={sample['index']} label={sample['ground_truth']} "
            f"original={sample['original_prediction']} llm={llm_prediction} status={parse_status}"
        )
        time.sleep(sleep_seconds)

    write_json(output_dir / "llm_judgments.json", judgment_rows)
    write_jsonl(output_dir / "llm_judgments.jsonl", judgment_rows)
    write_csv(output_dir / "llm_predictions.csv", csv_rows)
    write_json(
        output_dir / "llm_summary.json",
        {
            "dataset_id": dataset_id,
            "input_json": str(input_json),
            "prompt_file": str(prompt_file),
            "model": model,
            "total_samples": len(positive_samples),
            "parsed_predictions": sum(row["llm_prediction"] in (0, 1) for row in judgment_rows),
            "output_dir": str(output_dir),
        },
    )

    print(f"dataset_id={dataset_id}")
    print(f"total_samples={len(positive_samples)}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
