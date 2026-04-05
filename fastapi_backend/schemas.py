from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


ClassicalAction = Literal["train", "test"]
LLMAction = Literal["finetune", "inference"]


class ClassicalJobRequest(BaseModel):
    action: ClassicalAction
    dataset_name: str = Field(..., min_length=1)
    model_name: str = Field(..., min_length=1)
    length: str = Field(..., min_length=1)
    cuda: str = "0"


class ImbalanceClassicalJobRequest(BaseModel):
    action: ClassicalAction
    dataset_name: str = Field(..., min_length=1)
    model_name: str = Field(..., min_length=1)
    pos_ratio: str = Field(..., min_length=1)
    cuda: str = "0"


class LLMJobRequest(BaseModel):
    action: LLMAction
    dataset_name: str = Field(..., min_length=1)
    model_name: str = Field(..., min_length=1)
    length: str = Field(..., min_length=1)
    batch_size: Optional[int] = Field(default=None, ge=1)
    cuda: str = "0"


class ImbalanceLLMJobRequest(BaseModel):
    action: LLMAction
    dataset_name: str = Field(..., min_length=1)
    model_name: str = Field(..., min_length=1)
    pos_ratio: str = Field(..., min_length=1)
    cuda: str = "0"


class AblationJobRequest(BaseModel):
    action: LLMAction
    dataset_name: str = Field(..., min_length=1)
    model_name: str = Field(..., min_length=1)
    r: int = Field(..., ge=1)
    alpha: int = Field(..., ge=1)
    cuda: str = "0"


class ToGraphJobRequest(BaseModel):
    dataset_name: str = Field(..., min_length=1)
    length: str = Field(..., min_length=1)


class JobResponse(BaseModel):
    job_id: str
    job_type: str
    script_name: str
    status: str
    command: list[str]
    pid: Optional[int]
    created_at: datetime
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    return_code: Optional[int]
    output_dir: Optional[str]
    log_file: Optional[str]
    result_csv: Optional[str]
    params: dict[str, str]
    error_message: Optional[str]


class JobLogResponse(BaseModel):
    job_id: str
    log_file: Optional[str]
    content: str


class OutputEntry(BaseModel):
    name: str
    relative_path: str
    entry_type: Literal["file", "directory"]
    size: Optional[int]


class OutputListResponse(BaseModel):
    base_dir: str
    relative_path: str
    entries: list[OutputEntry]


class OutputTextResponse(BaseModel):
    base_dir: str
    relative_path: str
    content: str
