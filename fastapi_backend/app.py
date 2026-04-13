from __future__ import annotations

import mimetypes
import os
from pathlib import Path
from typing import Iterable, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from fastapi_backend.job_runner import JobManager, REPO_ROOT
from fastapi_backend.schemas import (
    AblationJobRequest,
    ClassicalJobRequest,
    FileOption,
    ImbalanceClassicalJobRequest,
    ImbalanceLLMJobRequest,
    JobLogResponse,
    JobResponse,
    LLMTestEnsembleJobRequest,
    LLMTestExtractJobRequest,
    LLMTestJudgeJobRequest,
    LLMTestMetricsJobRequest,
    LLMTestOptionsResponse,
    LLMTestReviewJobRequest,
    LLMJobRequest,
    OutputEntry,
    OutputListResponse,
    OutputTextResponse,
    ToGraphJobRequest,
)

app = FastAPI(
    title="LLM4CVD FastAPI Backend",
    version="0.1.0",
    description="HTTP API for launching the existing training and inference scripts in this repository.",
)

origins = [item.strip() for item in os.getenv("ALLOWED_ORIGINS", "*").split(",") if item.strip()]
allow_credentials = origins != ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

job_manager = JobManager()
OUTPUTS_ROOT = REPO_ROOT / "outputs"
LLM_TEST_ROOT = REPO_ROOT / "LLM_TEST"
LLM_TEST_INTERMEDIATE_ROOT = LLM_TEST_ROOT / "intermediate"
LLM_TEST_OUTPUT_ROOT = LLM_TEST_ROOT / "output"
LLM_TEST_ALLOWED_ROOTS = {
    "intermediate": LLM_TEST_INTERMEDIATE_ROOT,
    "output": LLM_TEST_OUTPUT_ROOT,
}

GRAPH_MODELS = {"Devign", "ReGVD", "GraphCodeBERT", "CodeBERT", "UniXcoder"}
LLM_MODELS = {"llama2", "llama3", "llama3.1", "codellama","llama3.2"}


def _job_response(record) -> JobResponse:
    return JobResponse(**record.as_dict())


def _assert_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{label} not found: {path}")


def _tail_text(path: Path, lines: int) -> str:
    if not path.exists():
        return ""
    content = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(content[-lines:])


def _resolve_outputs_path(relative_path: str = "") -> Path:
    candidate = (OUTPUTS_ROOT / relative_path).resolve()
    try:
        candidate.relative_to(OUTPUTS_ROOT.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid outputs path.") from exc
    return candidate


def _resolve_repo_path(path_value: str, label: str, *, allow_empty: bool = False) -> Optional[Path]:
    cleaned = path_value.strip()
    if not cleaned:
        if allow_empty:
            return None
        raise HTTPException(status_code=400, detail=f"{label} cannot be empty.")

    candidate = Path(cleaned)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    candidate = candidate.resolve()

    try:
        candidate.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"{label} must stay inside the repository.") from exc
    return candidate


def _resolve_llm_test_path(root_key: str, relative_path: str = "") -> Path:
    base_root = LLM_TEST_ALLOWED_ROOTS.get(root_key)
    if base_root is None:
        raise HTTPException(status_code=400, detail="Unsupported LLM_TEST root.")

    candidate = (base_root / relative_path).resolve()
    try:
        candidate.relative_to(base_root.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid LLM_TEST path.") from exc
    return candidate


def _read_text_file(path: Path, max_chars: int) -> str:
    content = path.read_text(encoding="utf-8", errors="ignore")
    if len(content) > max_chars:
        return content[:max_chars]
    return content


def _to_relative_repo_path(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve()))


def _build_file_option(path: Path) -> FileOption:
    relative_path = _to_relative_repo_path(path)
    return FileOption(label=path.parent.name + "/" + path.name, path=relative_path)


def _scan_glob_options(base_root: Path, pattern: str) -> list[FileOption]:
    if not base_root.exists():
        return []
    return [_build_file_option(path) for path in sorted(base_root.glob(pattern))]


def _scan_dir_options(base_root: Path, child_name: str) -> list[FileOption]:
    if not base_root.exists():
        return []

    options: list[FileOption] = []
    for path in sorted(base_root.glob(f"**/{child_name}")):
        if path.parent.is_dir():
            options.append(
                FileOption(
                    label=path.parent.name,
                    path=_to_relative_repo_path(path.parent),
                )
            )
    return options


def _list_directory(root_path: Path, target: Path) -> OutputListResponse:
    entries = []
    for item in sorted(target.iterdir(), key=lambda path: (not path.is_dir(), path.name.lower())):
        resolved = item.resolve()
        entries.append(
            OutputEntry(
                name=item.name,
                relative_path=str(resolved.relative_to(root_path.resolve())),
                entry_type="directory" if item.is_dir() else "file",
                size=None if item.is_dir() else item.stat().st_size,
            )
        )

    normalized_relative = "" if target == root_path else str(target.relative_to(root_path.resolve()))
    return OutputListResponse(
        base_dir=str(root_path),
        relative_path=normalized_relative,
        entries=entries,
    )


def _scan_data_directories() -> dict[str, list[str]]:
    data_root = REPO_ROOT / "data"
    regular = []
    imbalance = []
    if data_root.exists():
        for child in sorted(item.name for item in data_root.iterdir() if item.is_dir()):
            if child.endswith("_subsampled"):
                imbalance.append(child)
            else:
                regular.append(child)
    return {"regular": regular, "imbalance": imbalance}


def _collect_suffixes(directory: Path, prefix: str, suffix: str) -> list[str]:
    if not directory.exists():
        return []
    values = set()
    for file_path in directory.glob(f"{prefix}*{suffix}"):
        stem = file_path.name
        if not stem.startswith(prefix) or not stem.endswith(suffix):
            continue
        values.add(stem[len(prefix) : -len(suffix)])
    return sorted(values)


def _dataset_prefix_from_directory(directory: Path) -> str:
    dataset_name = directory.parent.name
    if dataset_name.endswith("_subsampled"):
        dataset_name = dataset_name[: -len("_subsampled")]
    return f"{dataset_name}_"


def _collect_imbalance_parts(directory: Path) -> tuple[list[str], list[str]]:
    lengths = set()
    ratios = set()
    if not directory.exists():
        return [], []

    prefix = _dataset_prefix_from_directory(directory)
    suffix = "_train.json"
    for item in _collect_suffixes(directory, prefix, suffix):
        parts = item.split("_")
        if len(parts) >= 2 and "-" in parts[0]:
            lengths.add(parts[0])
            ratios.add("_".join(parts[1:]))
        elif item:
            ratios.add(item)
    return sorted(lengths), sorted(ratios)


def _stringify_params(payload: dict[str, object]) -> dict[str, str]:
    return {key: str(value) for key, value in payload.items() if value is not None}


def _launch_command_job(
    *,
    job_type: str,
    script_name: str,
    command: list[str],
    output_dir: Path | None,
    log_file: Path | None,
    result_csv: Path | None,
    params: dict[str, object],
):
    record = job_manager.start_job(
        job_type=job_type,
        script_name=script_name,
        command=command,
        params=_stringify_params(params),
        output_dir=output_dir,
        log_file=log_file,
        result_csv=result_csv,
        working_dir=REPO_ROOT,
    )
    return _job_response(record)


def _launch_job(
    *,
    job_type: str,
    script_name: str,
    args: Iterable[object],
    output_dir: Path | None,
    log_file: Path | None,
    result_csv: Path | None,
    params: dict[str, object],
):
    script_path = REPO_ROOT / "scripts" / script_name
    _assert_exists(script_path, "script")
    command = ["bash", str(script_path), *[str(arg) for arg in args]]
    record = job_manager.start_job(
        job_type=job_type,
        script_name=script_name,
        command=command,
        params=_stringify_params(params),
        output_dir=output_dir,
        log_file=log_file,
        result_csv=result_csv,
        working_dir=REPO_ROOT,
    )
    return _job_response(record)


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/outputs", response_model=OutputListResponse)
def list_outputs(relative_path: str = Query(default="")) -> OutputListResponse:
    target = _resolve_outputs_path(relative_path)
    _assert_exists(target, "outputs path")
    if not target.is_dir():
        raise HTTPException(status_code=400, detail="Target path is not a directory.")
    return _list_directory(OUTPUTS_ROOT, target)


@app.get("/api/outputs/text", response_model=OutputTextResponse)
def read_output_text(
    relative_path: str = Query(...),
    max_chars: int = Query(default=200000, ge=1, le=1000000),
) -> OutputTextResponse:
    target = _resolve_outputs_path(relative_path)
    _assert_exists(target, "output file")
    if not target.is_file():
        raise HTTPException(status_code=400, detail="Target path is not a file.")

    content = _read_text_file(target, max_chars=max_chars)
    return OutputTextResponse(
        base_dir=str(OUTPUTS_ROOT),
        relative_path=str(target.relative_to(OUTPUTS_ROOT.resolve())),
        content=content,
    )


@app.get("/api/outputs/file")
def get_output_file(relative_path: str = Query(...)) -> FileResponse:
    target = _resolve_outputs_path(relative_path)
    _assert_exists(target, "output file")
    if not target.is_file():
        raise HTTPException(status_code=400, detail="Target path is not a file.")

    media_type, _ = mimetypes.guess_type(str(target))
    return FileResponse(target, media_type=media_type or "application/octet-stream", filename=target.name)


@app.get("/api/meta/options")
def get_options() -> dict[str, object]:
    datasets = _scan_data_directories()
    options = {
        "graph_models": sorted(GRAPH_MODELS),
        "llm_models": sorted(LLM_MODELS),
        "datasets": datasets,
        "regular_lengths": [],
        "imbalance_lengths": [],
        "imbalance_ratios": [],
    }

    regular_dirs = [REPO_ROOT / "data" / name / "alpaca" for name in datasets["regular"]]
    imbalance_dirs = [REPO_ROOT / "data" / name / "alpaca" for name in datasets["imbalance"]]

    if regular_dirs:
        options["regular_lengths"] = sorted(
            {
                item
                for directory in regular_dirs
                for item in _collect_suffixes(directory, _dataset_prefix_from_directory(directory), "_train.json")
                if item
            }
        )
    if imbalance_dirs:
        imbalance_lengths = set()
        imbalance_ratios = set()
        for directory in imbalance_dirs:
            lengths, ratios = _collect_imbalance_parts(directory)
            imbalance_lengths.update(lengths)
            imbalance_ratios.update(ratios)
        options["imbalance_lengths"] = sorted(imbalance_lengths)
        options["imbalance_ratios"] = sorted(imbalance_ratios)

    return options


@app.get("/api/meta/llm-test-options", response_model=LLMTestOptionsResponse)
def get_llm_test_options() -> LLMTestOptionsResponse:
    prompt_root = LLM_TEST_ROOT / "Prompt"
    prompt_paths = sorted(prompt_root.glob("*.txt")) if prompt_root.exists() else []
    prompt_files = [str(path.relative_to(REPO_ROOT)) for path in prompt_paths]

    config_files = [
        str(path.relative_to(REPO_ROOT))
        for path in [LLM_TEST_ROOT / "exp.yaml", LLM_TEST_ROOT / ".env", LLM_TEST_ROOT / ".env.last_run"]
        if path.exists()
    ]
    env_files = [path for path in config_files if path.endswith(".env") or path.endswith(".env.last_run")]

    return LLMTestOptionsResponse(
        config_files=config_files,
        env_files=env_files,
        prompt_files=prompt_files,
        review_scripts=[
            "LLM_TEST/extract_positive_samples.py",
            "LLM_TEST/llm_api_judge.py",
            "LLM_TEST/recompute_metrics.py",
            "LLM_TEST/run_review.sh",
            "LLM_TEST/ensemble_vote.py",
        ],
        ensemble_strategies=["any", "majority", "threshold", "weighted"],
        results_csv_options=_scan_glob_options(OUTPUTS_ROOT, "**/results.csv"),
        input_json_options=_scan_glob_options(LLM_TEST_INTERMEDIATE_ROOT, "**/positive_samples.json"),
        llm_predictions_csv_options=_scan_glob_options(LLM_TEST_OUTPUT_ROOT, "**/llm_predictions.csv"),
        metrics_json_options=_scan_glob_options(LLM_TEST_OUTPUT_ROOT, "**/metrics.json"),
        llm_summary_options=_scan_glob_options(LLM_TEST_OUTPUT_ROOT, "**/llm_summary.json"),
        intermediate_dir_options=_scan_dir_options(LLM_TEST_INTERMEDIATE_ROOT, "positive_samples.json"),
        output_dir_options=_scan_dir_options(LLM_TEST_OUTPUT_ROOT, "llm_predictions.csv"),
    )


@app.get("/api/llm-test/files", response_model=OutputListResponse)
def list_llm_test_files(
    root: str = Query(..., pattern="^(intermediate|output)$"),
    relative_path: str = Query(default=""),
) -> OutputListResponse:
    target = _resolve_llm_test_path(root, relative_path)
    _assert_exists(target, "LLM_TEST path")
    if not target.is_dir():
        raise HTTPException(status_code=400, detail="Target path is not a directory.")
    return _list_directory(LLM_TEST_ALLOWED_ROOTS[root], target)


@app.get("/api/llm-test/files/text", response_model=OutputTextResponse)
def read_llm_test_text(
    root: str = Query(..., pattern="^(intermediate|output)$"),
    relative_path: str = Query(...),
    max_chars: int = Query(default=200000, ge=1, le=1000000),
) -> OutputTextResponse:
    target = _resolve_llm_test_path(root, relative_path)
    _assert_exists(target, "LLM_TEST file")
    if not target.is_file():
        raise HTTPException(status_code=400, detail="Target path is not a file.")

    base_root = LLM_TEST_ALLOWED_ROOTS[root]
    content = _read_text_file(target, max_chars=max_chars)
    return OutputTextResponse(
        base_dir=str(base_root),
        relative_path=str(target.relative_to(base_root.resolve())),
        content=content,
    )


@app.get("/api/llm-test/files/file")
def get_llm_test_file(
    root: str = Query(..., pattern="^(intermediate|output)$"),
    relative_path: str = Query(...),
) -> FileResponse:
    target = _resolve_llm_test_path(root, relative_path)
    _assert_exists(target, "LLM_TEST file")
    if not target.is_file():
        raise HTTPException(status_code=400, detail="Target path is not a file.")

    media_type, _ = mimetypes.guess_type(str(target))
    return FileResponse(target, media_type=media_type or "application/octet-stream", filename=target.name)


@app.post("/api/jobs/classical", response_model=JobResponse)
def create_classical_job(payload: ClassicalJobRequest) -> JobResponse:
    if payload.model_name not in GRAPH_MODELS:
        raise HTTPException(status_code=400, detail="Unsupported classical model.")
    script_name = "train.sh" if payload.action == "train" else "test.sh"
    output_dir = REPO_ROOT / "outputs" / payload.model_name / f"{payload.dataset_name}_{payload.length}"
    log_name = f"{payload.action}_{payload.model_name}_{payload.dataset_name}_{payload.length}.log"
    if payload.model_name == "Devign":
        log_path = output_dir / log_name
    else:
        log_path = output_dir / log_name
    return _launch_job(
        job_type="classical",
        script_name=script_name,
        args=[payload.dataset_name, payload.model_name, payload.length, payload.cuda],
        output_dir=output_dir,
        log_file=log_path,
        result_csv=output_dir / "results.csv",
        params=payload.model_dump(),
    )


@app.post("/api/jobs/classical-imbalance", response_model=JobResponse)
def create_imbalance_classical_job(payload: ImbalanceClassicalJobRequest) -> JobResponse:
    if payload.model_name not in GRAPH_MODELS:
        raise HTTPException(status_code=400, detail="Unsupported classical model.")
    script_name = "train_imbalance.sh" if payload.action == "train" else "test_imbalance.sh"
    output_dir = REPO_ROOT / "outputs" / f"{payload.model_name}_imbalance" / f"{payload.dataset_name}_{payload.length}_{payload.pos_ratio}"
    log_path = output_dir / f"{payload.action}_{payload.model_name}_{payload.dataset_name}_{payload.length}_{payload.pos_ratio}.log"
    return _launch_job(
        job_type="classical_imbalance",
        script_name=script_name,
        args=[payload.dataset_name, payload.model_name, payload.length, payload.pos_ratio, payload.cuda],
        output_dir=output_dir,
        log_file=log_path,
        result_csv=output_dir / "results.csv",
        params=payload.model_dump(),
    )


@app.post("/api/jobs/llm", response_model=JobResponse)
def create_llm_job(payload: LLMJobRequest) -> JobResponse:
    if payload.model_name not in LLM_MODELS:
        raise HTTPException(status_code=400, detail="Unsupported LLM model.")
    if payload.action == "finetune" and payload.batch_size is None:
        raise HTTPException(status_code=400, detail="batch_size is required for finetune.")
    script_name = "finetune.sh" if payload.action == "finetune" else "inference.sh"
    output_dir = REPO_ROOT / "outputs" / f"{payload.model_name}_lora" / f"{payload.dataset_name}_{payload.length}"
    log_prefix = "finetuning" if payload.action == "finetune" else "inference"
    log_path = output_dir / f"{log_prefix}_{payload.model_name}_lora_{payload.dataset_name}_{payload.length}.log"
    args = [payload.dataset_name, payload.model_name, payload.length]
    if payload.action == "finetune":
        args.append(payload.batch_size)
    args.append(payload.cuda)
    return _launch_job(
        job_type="llm",
        script_name=script_name,
        args=args,
        output_dir=output_dir,
        log_file=log_path,
        result_csv=output_dir / "results.csv",
        params=payload.model_dump(),
    )


@app.post("/api/jobs/llm-imbalance", response_model=JobResponse)
def create_imbalance_llm_job(payload: ImbalanceLLMJobRequest) -> JobResponse:
    if payload.model_name not in LLM_MODELS:
        raise HTTPException(status_code=400, detail="Unsupported LLM model.")
    script_name = "finetune_imbalance.sh" if payload.action == "finetune" else "inference_imbalance.sh"
    output_dir = REPO_ROOT / "outputs" / f"{payload.model_name}_lora_imbalance" / f"{payload.dataset_name}_{payload.pos_ratio}"
    log_prefix = "finetuning" if payload.action == "finetune" else "inference"
    log_path = output_dir / f"{log_prefix}_{payload.model_name}_lora_{payload.dataset_name}_{payload.pos_ratio}.log"
    return _launch_job(
        job_type="llm_imbalance",
        script_name=script_name,
        args=[payload.dataset_name, payload.model_name, payload.pos_ratio, payload.cuda],
        output_dir=output_dir,
        log_file=log_path,
        result_csv=output_dir / "results.csv",
        params=payload.model_dump(),
    )


@app.post("/api/jobs/ablation", response_model=JobResponse)
def create_ablation_job(payload: AblationJobRequest) -> JobResponse:
    if payload.model_name not in LLM_MODELS:
        raise HTTPException(status_code=400, detail="Unsupported LLM model.")
    script_name = "finetune_ablation.sh" if payload.action == "finetune" else "inference_ablation.sh"
    suffix = f"{payload.dataset_name}_{payload.r}_{payload.alpha}"
    output_dir = REPO_ROOT / "outputs" / f"{payload.model_name}_lora_ablation" / suffix
    log_suffix = f"{payload.model_name}_lora_ablation_{payload.dataset_name}_{payload.r}_{payload.alpha}.log"
    log_name = f"finetuning_{log_suffix}" if payload.action == "finetune" else f"inference_{log_suffix}"
    return _launch_job(
        job_type="ablation",
        script_name=script_name,
        args=[payload.dataset_name, payload.model_name, payload.r, payload.alpha, payload.cuda],
        output_dir=output_dir,
        log_file=output_dir / log_name,
        result_csv=output_dir / "results.csv",
        params=payload.model_dump(),
    )


@app.post("/api/jobs/to-graph", response_model=JobResponse)
def create_to_graph_job(payload: ToGraphJobRequest) -> JobResponse:
    output_dir = REPO_ROOT / "data" / payload.dataset_name / "graph"
    return _launch_job(
        job_type="to_graph",
        script_name="to_graph.sh",
        args=[payload.dataset_name, payload.length],
        output_dir=output_dir,
        log_file=None,
        result_csv=None,
        params=payload.model_dump(),
    )


@app.post("/api/jobs/llm-test/extract", response_model=JobResponse)
def create_llm_test_extract_job(payload: LLMTestExtractJobRequest) -> JobResponse:
    results_csv = _resolve_repo_path(payload.results_csv, "results_csv")
    data_root = _resolve_repo_path(payload.data_root, "data_root")
    output_root = _resolve_repo_path(payload.output_root, "output_root")
    data_json = _resolve_repo_path(payload.data_json, "data_json", allow_empty=True) if payload.data_json else None
    output_subdir = payload.output_subdir or results_csv.parent.name
    output_dir = output_root / output_subdir
    log_file = output_dir / "extract_positive_samples.log"

    command = [
        "python3",
        str(LLM_TEST_ROOT / "extract_positive_samples.py"),
        "--config",
        str(_resolve_repo_path(payload.config, "config")),
        "--env_file",
        str(_resolve_repo_path(payload.env_file, "env_file")),
        "--results_csv",
        str(results_csv),
        "--data_root",
        str(data_root),
        "--output_root",
        str(output_root),
        "--output_subdir",
        output_subdir,
        "--prediction_value",
        str(payload.prediction_value),
    ]
    if data_json is not None:
        command.extend(["--data_json", str(data_json)])
    if payload.min_prob is not None:
        command.extend(["--min_prob", str(payload.min_prob)])
    if payload.max_prob is not None:
        command.extend(["--max_prob", str(payload.max_prob)])

    return _launch_command_job(
        job_type="llm_test_extract",
        script_name="extract_positive_samples.py",
        command=command,
        output_dir=output_dir,
        log_file=log_file,
        result_csv=output_dir / "positive_samples.json",
        params=payload.model_dump(),
    )


@app.post("/api/jobs/llm-test/judge", response_model=JobResponse)
def create_llm_test_judge_job(payload: LLMTestJudgeJobRequest) -> JobResponse:
    input_json = _resolve_repo_path(payload.input_json, "input_json")
    prompt_file = _resolve_repo_path(payload.prompt_file, "prompt_file")
    output_root = _resolve_repo_path(payload.output_root, "output_root")
    output_name = payload.output_name or input_json.parent.name
    output_dir = output_root / output_name
    log_file = output_dir / "llm_api_judge.log"

    command = [
        "python3",
        str(LLM_TEST_ROOT / "llm_api_judge.py"),
        "--config",
        str(_resolve_repo_path(payload.config, "config")),
        "--env_file",
        str(_resolve_repo_path(payload.env_file, "env_file")),
        "--input_json",
        str(input_json),
        "--prompt_file",
        str(prompt_file),
        "--output_root",
        str(output_root),
        "--workers",
        str(payload.workers),
        "--temperature",
        str(payload.temperature),
        "--max_tokens",
        str(payload.max_tokens),
        "--timeout",
        str(payload.timeout),
        "--retries",
        str(payload.retries),
        "--sleep_seconds",
        str(payload.sleep_seconds),
    ]
    if payload.output_name:
        command.extend(["--output_name", payload.output_name])
    if payload.output_by_prompt_version:
        command.append("--output_by_prompt_version")
    if payload.limit is not None:
        command.extend(["--limit", str(payload.limit)])
    if payload.model:
        command.extend(["--model", payload.model])
    if payload.api_base:
        command.extend(["--api_base", payload.api_base])
    if payload.api_key:
        command.extend(["--api_key", payload.api_key])
    if payload.fail_fast_on_error:
        command.append("--fail_fast_on_error")

    return _launch_command_job(
        job_type="llm_test_judge",
        script_name="llm_api_judge.py",
        command=command,
        output_dir=output_dir,
        log_file=log_file,
        result_csv=output_dir / "llm_predictions.csv",
        params=payload.model_dump(exclude={"api_key"}, exclude_none=True),
    )


@app.post("/api/jobs/llm-test/metrics", response_model=JobResponse)
def create_llm_test_metrics_job(payload: LLMTestMetricsJobRequest) -> JobResponse:
    results_csv = _resolve_repo_path(payload.results_csv, "results_csv")
    llm_predictions_csv = _resolve_repo_path(payload.llm_predictions_csv, "llm_predictions_csv")
    output_dir = (
        _resolve_repo_path(payload.output_dir, "output_dir")
        if payload.output_dir
        else llm_predictions_csv.parent
    )
    log_file = output_dir / "recompute_metrics.log"

    command = [
        "python3",
        str(LLM_TEST_ROOT / "recompute_metrics.py"),
        "--config",
        str(_resolve_repo_path(payload.config, "config")),
        "--env_file",
        str(_resolve_repo_path(payload.env_file, "env_file")),
        "--results_csv",
        str(results_csv),
        "--llm_predictions_csv",
        str(llm_predictions_csv),
        "--output_dir",
        str(output_dir),
    ]

    return _launch_command_job(
        job_type="llm_test_metrics",
        script_name="recompute_metrics.py",
        command=command,
        output_dir=output_dir,
        log_file=log_file,
        result_csv=output_dir / "metrics.json",
        params=payload.model_dump(exclude_none=True),
    )


@app.post("/api/jobs/llm-test/review", response_model=JobResponse)
def create_llm_test_review_job(payload: LLMTestReviewJobRequest) -> JobResponse:
    config_path = _resolve_repo_path(payload.config, "config")
    env_path = _resolve_repo_path(payload.env, "env")
    output_root = _resolve_repo_path(payload.output_root, "output_root") if payload.output_root else None

    command = [
        "bash",
        str(LLM_TEST_ROOT / "run_review.sh"),
        "--result-model",
        payload.result_model,
        "--dataset",
        payload.dataset,
        "--prob-range",
        payload.prob_range,
        "--workers",
        str(payload.workers),
        "--config",
        str(config_path),
        "--env",
        str(env_path),
    ]
    if payload.limit is not None:
        command.extend(["--limit", str(payload.limit)])
    if payload.data_json:
        command.extend(["--data-json", str(_resolve_repo_path(payload.data_json, "data_json"))])
    if payload.prompt_file:
        command.extend(["--prompt-file", str(_resolve_repo_path(payload.prompt_file, "prompt_file"))])
    if output_root is not None:
        command.extend(["--output-root", str(output_root)])

    review_root = output_root if output_root is not None else LLM_TEST_OUTPUT_ROOT
    output_dir = review_root / f"{payload.dataset}_{payload.result_model}"
    log_file = output_dir / "run_review.log"

    return _launch_command_job(
        job_type="llm_test_review",
        script_name="run_review.sh",
        command=command,
        output_dir=output_dir,
        log_file=log_file,
        result_csv=None,
        params=payload.model_dump(exclude_none=True),
    )


@app.post("/api/jobs/llm-test/ensemble", response_model=JobResponse)
def create_llm_test_ensemble_job(payload: LLMTestEnsembleJobRequest) -> JobResponse:
    output_dir = (
        _resolve_repo_path(payload.output_dir, "output_dir")
        if payload.output_dir
        else _resolve_repo_path(payload.inputs[0].path, "inputs[0].path").parent
    )
    log_file = output_dir / f"{payload.output_prefix}_ensemble.log"

    command = [
        "python3",
        str(LLM_TEST_ROOT / "ensemble_vote.py"),
        "--inputs",
    ]
    for item in payload.inputs:
        input_path = _resolve_repo_path(item.path, f"inputs[{item.name}]")
        command.append(f"{item.name}={input_path}")
    command.extend(["--strategy", payload.strategy, "--output_dir", str(output_dir), "--output_prefix", payload.output_prefix])
    if payload.threshold is not None:
        command.extend(["--threshold", str(payload.threshold)])
    if payload.intersection_only:
        command.append("--intersection_only")
    if payload.weights:
        command.append("--weights")
        for name, weight in payload.weights.items():
            command.append(f"{name}={weight}")

    return _launch_command_job(
        job_type="llm_test_ensemble",
        script_name="ensemble_vote.py",
        command=command,
        output_dir=output_dir,
        log_file=log_file,
        result_csv=output_dir / f"{payload.output_prefix}_metrics.json",
        params=payload.model_dump(exclude_none=True),
    )


@app.get("/api/jobs", response_model=list[JobResponse])
def list_jobs() -> list[JobResponse]:
    return [_job_response(record) for record in job_manager.list_jobs()]


@app.get("/api/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str) -> JobResponse:
    record = job_manager.get_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return _job_response(record)


@app.get("/api/jobs/{job_id}/log", response_model=JobLogResponse)
def get_job_log(job_id: str, lines: int = Query(default=200, ge=1, le=2000)) -> JobLogResponse:
    record = job_manager.get_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    log_file = Path(record.log_file) if record.log_file else None
    content = _tail_text(log_file, lines) if log_file else ""
    return JobLogResponse(job_id=job_id, log_file=record.log_file, content=content)


@app.post("/api/jobs/{job_id}/stop", response_model=JobResponse)
def stop_job(job_id: str) -> JobResponse:
    record = job_manager.stop_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return _job_response(record)
