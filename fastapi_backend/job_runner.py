from __future__ import annotations

import os
import subprocess
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class JobRecord:
    job_id: str
    job_type: str
    script_name: str
    command: list[str]
    params: dict[str, str]
    output_dir: Optional[str] = None
    log_file: Optional[str] = None
    result_csv: Optional[str] = None
    status: str = "queued"
    pid: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    return_code: Optional[int] = None
    error_message: Optional[str] = None
    process: Optional[subprocess.Popen] = field(default=None, repr=False, compare=False)

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload.pop("process", None)
        return payload


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def start_job(
        self,
        *,
        job_type: str,
        script_name: str,
        command: list[str],
        params: dict[str, str],
        output_dir: Optional[Path] = None,
        log_file: Optional[Path] = None,
        result_csv: Optional[Path] = None,
    ) -> JobRecord:
        job_id = uuid.uuid4().hex
        record = JobRecord(
            job_id=job_id,
            job_type=job_type,
            script_name=script_name,
            command=command,
            params=params,
            output_dir=str(output_dir) if output_dir else None,
            log_file=str(log_file) if log_file else None,
            result_csv=str(result_csv) if result_csv else None,
        )

        try:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            record.status = "failed"
            record.error_message = str(exc)
            record.finished_at = datetime.utcnow()
            with self._lock:
                self._jobs[job_id] = record
            return record

        record.process = process
        record.pid = process.pid
        record.status = "running"
        record.started_at = datetime.utcnow()

        with self._lock:
            self._jobs[job_id] = record

        watcher = threading.Thread(target=self._watch_job, args=(job_id,), daemon=True)
        watcher.start()
        return record

    def _watch_job(self, job_id: str) -> None:
        record = self.get_job(job_id)
        if record is None or record.process is None:
            return

        return_code = record.process.wait()
        with self._lock:
            current = self._jobs.get(job_id)
            if current is None:
                return
            current.return_code = return_code
            current.finished_at = datetime.utcnow()
            if current.status == "stopped":
                return
            current.status = "completed" if return_code == 0 else "failed"

    def list_jobs(self) -> list[JobRecord]:
        with self._lock:
            jobs = list(self._jobs.values())

        for job in jobs:
            self._sync_status(job)
        return sorted(jobs, key=lambda item: item.created_at, reverse=True)

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is not None:
            self._sync_status(job)
        return job

    def stop_job(self, job_id: str) -> Optional[JobRecord]:
        record = self.get_job(job_id)
        if record is None or record.process is None:
            return record

        if record.process.poll() is None:
            try:
                os.killpg(os.getpgid(record.process.pid), 15)
            except ProcessLookupError:
                pass

        with self._lock:
            current = self._jobs.get(job_id)
            if current is None:
                return record
            current.status = "stopped"
            current.finished_at = datetime.utcnow()
            current.return_code = current.process.poll()
            return current

    def _sync_status(self, record: JobRecord) -> None:
        if record.process is None:
            return

        return_code = record.process.poll()
        if return_code is None:
            return

        with self._lock:
            current = self._jobs.get(record.job_id)
            if current is None or current.status in {"completed", "failed", "stopped"}:
                return
            current.return_code = return_code
            current.finished_at = current.finished_at or datetime.utcnow()
            current.status = "completed" if return_code == 0 else "failed"

