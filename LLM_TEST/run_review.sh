#!/bin/bash
set -euo pipefail

CONFIG_PATH="${1:-LLM_TEST/exp.yaml}"
ENV_PATH="${2:-LLM_TEST/.env}"
LIMIT="${3:-100}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Config file not found: ${CONFIG_PATH}"
    exit 1
fi

if [[ ! -f "${ENV_PATH}" ]]; then
    echo "Env file not found: ${ENV_PATH}"
    exit 1
fi

echo "Starting LLM review"
echo "config=${CONFIG_PATH}"
echo "env=${ENV_PATH}"
echo "limit=${LIMIT}"

python3 LLM_TEST/llm_api_judge.py \
    --config "${CONFIG_PATH}" \
    --env_file "${ENV_PATH}" \
    --limit "${LIMIT}" \
    --output_by_prompt_version

python3 LLM_TEST/recompute_metrics.py \
    --config "${CONFIG_PATH}" \
    --env_file "${ENV_PATH}"

echo "Completed."
