# LLM_TEST API

`LLM_TEST` 目录中的复审流程已经通过 `fastapi_backend` 暴露为 HTTP 接口，适合前端或脚本按“异步任务”方式调用。

服务启动后可在 `http://127.0.0.1:8000/docs` 查看 Swagger，也可以直接调用下面这些接口。

## 使用前提

在仓库根目录启动 FastAPI：

```bash
pip install -r requirements.txt
./scripts/run_fastapi.sh 0.0.0.0 8000
```

建议先准备好：

- `LLM_TEST/exp.yaml`
- `LLM_TEST/.env`
- `LLM_TEST/Prompt/*.txt`
- 已存在的小模型推理结果，例如 `outputs/<result_model>/<dataset>/results.csv`

## 调用方式

所有 `LLM_TEST` 接口都会立即返回一个作业对象，核心字段与其他训练接口一致：

- `job_id`: 任务 ID
- `status`: `queued` / `running` / `completed` / `failed` / `stopped`
- `log_file`: 后端保存的日志文件
- `result_csv`: 该任务最核心的输出文件
- `output_dir`: 该任务输出目录

轮询任务状态：

```bash
curl http://127.0.0.1:8000/api/jobs/<job_id>
curl "http://127.0.0.1:8000/api/jobs/<job_id>/log?lines=200"
```

## 元信息接口

### `GET /api/meta/llm-test-options`

返回 `LLM_TEST` 常用配置入口，包含：

- `config_files`
- `env_files`
- `prompt_files`
- `review_scripts`
- `ensemble_strategies`
- `results_csv_options`
- `input_json_options`
- `llm_predictions_csv_options`
- `metrics_json_options`
- `llm_summary_options`
- `intermediate_dir_options`
- `output_dir_options`

示例：

```bash
curl http://127.0.0.1:8000/api/meta/llm-test-options
```

其中几类最适合前端直接做下拉选择：

- `input_json_options`: 可直接用于 `judge` 接口里的 `input_json`
- `prompt_files`: 可直接用于 `prompt_file`
- `results_csv_options`: 可直接用于 `extract` 或 `metrics` 里的 `results_csv`
- `llm_predictions_csv_options`: 可直接用于 `metrics` 里的 `llm_predictions_csv`

返回项结构示例：

```json
{
  "label": "bigvul_1_llama3.2_lora_imbalance_positive_0.5_0.9/positive_samples.json",
  "path": "LLM_TEST/intermediate/bigvul_1_llama3.2_lora_imbalance_positive_0.5_0.9/positive_samples.json"
}
```

## 核心作业接口

### `POST /api/jobs/llm-test/extract`

用途：执行 `LLM_TEST/extract_positive_samples.py`，从 `results.csv` 中提取待复审样本。

最小请求体：

```json
{
  "results_csv": "outputs/llama3.2_lora_imbalance/bigvul_1/results.csv"
}
```

常用字段：

- `results_csv`: 原始推理结果 CSV
- `data_json`: 对应测试集 JSON，可省略并交给脚本自动推断
- `output_subdir`: 中间结果目录名
- `min_prob` / `max_prob`: 概率过滤区间

输出目录默认在 `LLM_TEST/intermediate/<output_subdir>/`，会生成：

- `positive_samples.json`
- `positive_indices.json`
- `summary.json`

### `POST /api/jobs/llm-test/judge`

用途：执行 `LLM_TEST/llm_api_judge.py`，调用 OpenAI 兼容接口复审抽取样本。

最小请求体：

```json
{
  "input_json": "LLM_TEST/intermediate/bigvul_1_llama3.2_lora_imbalance_positive_0.5_0.9/positive_samples.json",
  "prompt_file": "LLM_TEST/Prompt/CWE-119_0.5.txt"
}
```

常用字段：

- `model`: 覆盖 `.env` 中的 `LLM_MODEL`
- `api_base`: 覆盖 `.env` 中的 `API_BASE`
- `api_key`: 直接传入 API Key
- `output_name`: 输出子目录名
- `limit`: 只复审前 N 条
- `workers`: 并发请求数

输出目录默认在 `LLM_TEST/output/<output_name>/`，会生成：

- `llm_predictions.csv`
- `llm_judgments.json`
- `llm_summary.json`

### `POST /api/jobs/llm-test/metrics`

用途：执行 `LLM_TEST/recompute_metrics.py`，把 LLM 复审结果合回原始结果并重新计算指标。

最小请求体：

```json
{
  "results_csv": "outputs/llama3.2_lora_imbalance/bigvul_1/results.csv",
  "llm_predictions_csv": "LLM_TEST/output/bigvul_1_llama3.2_lora_imbalance_deepseek_CWE-119_0.5/llm_predictions.csv"
}
```

输出目录默认使用 `llm_predictions.csv` 所在目录，核心产物包括：

- `metrics.json`
- `metrics_first_100_llm_only.json`
- `merged_results.csv`
- `positive_case_details.csv`

### `POST /api/jobs/llm-test/review`

用途：执行 `LLM_TEST/run_review.sh` 的自动模式，一次性串联：

1. 提取正样本
2. API 复审
3. 重算指标

请求体示例：

```json
{
  "result_model": "llama3.2_lora_imbalance",
  "dataset": "bigvul_1",
  "prob_range": "0.5-0.9",
  "limit": 100,
  "workers": 2
}
```

可选覆盖项：

- `prompt_file`
- `data_json`
- `output_root`
- `config`
- `env`

这个接口最适合前端提供“一键复审”按钮。

### `POST /api/jobs/llm-test/ensemble`

用途：执行 `LLM_TEST/ensemble_vote.py`，对多个 `llm_predictions.csv` 做投票。

请求体示例：

```json
{
  "inputs": [
    {
      "name": "deepseek",
      "path": "LLM_TEST/output/deepseek/bigvul_tag/llm_predictions.csv"
    },
    {
      "name": "llama",
      "path": "LLM_TEST/output/llama3.2/bigvul_tag/llm_predictions.csv"
    }
  ],
  "strategy": "majority",
  "intersection_only": true,
  "output_dir": "LLM_TEST/output/ensemble_bigvul_tag",
  "output_prefix": "majority_2model"
}
```

支持的 `strategy`：

- `any`
- `majority`
- `threshold`
- `weighted`

如果使用 `weighted`，可通过 `weights` 传：

```json
{
  "weights": {
    "deepseek": 1.2,
    "llama": 0.8
  }
}
```

## LLM_TEST 产物浏览接口

为了方便前端直接浏览 `LLM_TEST/intermediate` 和 `LLM_TEST/output`，还提供了文件接口。

### `GET /api/llm-test/files`

查询参数：

- `root`: `intermediate` 或 `output`
- `relative_path`: 相对路径，默认空字符串

示例：

```bash
curl "http://127.0.0.1:8000/api/llm-test/files?root=output"
```

### `GET /api/llm-test/files/text`

读取文本文件内容：

```bash
curl "http://127.0.0.1:8000/api/llm-test/files/text?root=output&relative_path=some_run/metrics.json"
```

### `GET /api/llm-test/files/file`

直接下载文件：

```bash
curl -O "http://127.0.0.1:8000/api/llm-test/files/file?root=output&relative_path=some_run/llm_predictions.csv"
```

## 推荐前端接入顺序

如果前端要做完整操作台，推荐按这个顺序接：

1. 用 `GET /api/meta/llm-test-options` 拉取提示词和配置选项
2. 用 `POST /api/jobs/llm-test/review` 创建一键复审任务
3. 用 `GET /api/jobs/{job_id}` 和 `GET /api/jobs/{job_id}/log` 轮询状态和日志
4. 用 `GET /api/llm-test/files` 系列接口浏览中间结果和最终产物

## 路径约束

出于安全考虑，这些接口只接受仓库内部路径：

- 相对路径会自动按仓库根目录解析
- 绝对路径也必须位于当前仓库内
- 仓库外路径会被拒绝
