from __future__ import annotations

from pathlib import Path
from typing import Optional

from chunk_splite.build_cvefixes_astchunk import build_chunks, normalize_language, token_count

from fastapi_backend.job_runner import REPO_ROOT


DEFAULT_TOKENIZER_MODEL = REPO_ROOT / "model" / "Llama-3.2-1B"


def _resolve_tokenizer_model_path(path_value: Optional[str]) -> Optional[str]:
    if path_value is None:
        default_path = DEFAULT_TOKENIZER_MODEL
        return str(default_path) if default_path.exists() else None

    cleaned = path_value.strip()
    if not cleaned:
        return None

    candidate = Path(cleaned)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return str(candidate.resolve())


def run_code_chunking(
    *,
    code: str,
    language: Optional[str] = None,
    max_chars: int = 1800,
    max_tokens: int = 512,
    fallback_lines: int = 80,
    tokenizer_model_path: Optional[str] = None,
) -> dict[str, object]:
    resolved_tokenizer_model_path = _resolve_tokenizer_model_path(tokenizer_model_path)
    normalized_language = normalize_language(language)
    chunks, chunk_source = build_chunks(
        code=code,
        language=language,
        max_chars=max_chars,
        max_tokens=max_tokens,
        fallback_lines=fallback_lines,
        tokenizer_model_path=resolved_tokenizer_model_path,
    )

    chunk_payloads: list[dict[str, object]] = []
    for index, chunk in enumerate(chunks):
        chunk_payloads.append(
            {
                "index": index,
                "text": chunk.text,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "token_count": token_count(chunk.text, resolved_tokenizer_model_path),
            }
        )

    return {
        "language": language,
        "normalized_language": normalized_language,
        "chunk_source": chunk_source,
        "max_chars": max_chars,
        "max_tokens": max_tokens,
        "fallback_lines": fallback_lines,
        "tokenizer_model_path": resolved_tokenizer_model_path,
        "chunk_count": len(chunk_payloads),
        "chunks": chunk_payloads,
    }
