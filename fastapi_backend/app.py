from __future__ import annotations

import mimetypes
import os
from pathlib import Path
from typing import Iterable

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from fastapi_backend.job_runner import JobManager, REPO_ROOT
from fastapi_backend.schemas import (
    AblationJobRequest,
    ClassicalJobRequest,
    ImbalanceClassicalJobRequest,
    ImbalanceLLMJobRequest,
    JobLogResponse,
    JobResponse,
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

GRAPH_MODELS = {"Devign", "ReGVD", "GraphCodeBERT", "CodeBERT", "UniXcoder"}
LLM_MODELS = {"llama2", "llama3", "llama3.1", "codellama"}


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


def _read_text_file(path: Path, max_chars: int) -> str:
    content = path.read_text(encoding="utf-8", errors="ignore")
    if len(content) > max_chars:
        return content[:max_chars]
    return content


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


def _stringify_params(payload: dict[str, object]) -> dict[str, str]:
    return {key: str(value) for key, value in payload.items() if value is not None}


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

    entries = []
    for item in sorted(target.iterdir(), key=lambda path: (not path.is_dir(), path.name.lower())):
        resolved = item.resolve()
        entries.append(
            OutputEntry(
                name=item.name,
                relative_path=str(resolved.relative_to(OUTPUTS_ROOT.resolve())),
                entry_type="directory" if item.is_dir() else "file",
                size=None if item.is_dir() else item.stat().st_size,
            )
        )

    normalized_relative = "" if target == OUTPUTS_ROOT else str(target.relative_to(OUTPUTS_ROOT.resolve()))
    return OutputListResponse(
        base_dir=str(OUTPUTS_ROOT),
        relative_path=normalized_relative,
        entries=entries,
    )


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
        "imbalance_ratios": [],
    }

    regular_dirs = [REPO_ROOT / "data" / name / "alpaca" for name in datasets["regular"]]
    imbalance_dirs = [REPO_ROOT / "data" / name / "alpaca" for name in datasets["imbalance"]]

    if regular_dirs:
        options["regular_lengths"] = sorted(
            {
                item
                for directory in regular_dirs
                for item in _collect_suffixes(directory, f"{directory.parent.parent.name}_", "_train.json")
            }
        )
    if imbalance_dirs:
        options["imbalance_ratios"] = sorted(
            {
                item
                for directory in imbalance_dirs
                for item in _collect_suffixes(directory, f"{directory.parent.parent.name}_", "_train.json")
            }
        )

    return options


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
    output_dir = REPO_ROOT / "outputs" / f"{payload.model_name}_imbalance" / f"{payload.dataset_name}_1_{payload.pos_ratio}"
    log_path = output_dir / f"{payload.action}_{payload.model_name}_{payload.dataset_name}_1_{payload.pos_ratio}.log"
    return _launch_job(
        job_type="classical_imbalance",
        script_name=script_name,
        args=[payload.dataset_name, payload.model_name, payload.pos_ratio, payload.cuda],
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
