# FastAPI Backend

这个目录提供了一个面向前端的 FastAPI 后端，用来把仓库原本的 `scripts/*.sh` 能力暴露成 HTTP 接口。  
后端本身不直接重写训练逻辑，而是继续调用现有脚本，例如 `train.sh`、`test.sh`、`finetune.sh`、`inference.sh` 等。

除了训练和推理接口，现在也包含了 `LLM_TEST/` 复审工作流的接口封装。  
`LLM_TEST` 的专门接口说明见：[LLM_TEST/API_README.md](/home/zjr123/LLM4CVD-main/LLM_TEST/API_README.md)。

## 目录说明

- `app.py`: FastAPI 入口，定义所有路由
- `schemas.py`: 请求体和响应体的数据结构
- `job_runner.py`: 后台任务启动、状态维护、终止逻辑

## 启动方式

在仓库根目录执行：

```bash
pip install -r requirements.txt
./scripts/run_fastapi.sh 0.0.0.0 8000
```

启动后可访问：

- `http://127.0.0.1:8000/docs`：Swagger 文档页面
- `http://127.0.0.1:8000/health`：健康检查

如果前端本地开发端口需要跨域访问，可以配置：

```bash
ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:3000 ./scripts/run_fastapi.sh
```

## 设计思路

所有训练、测试、推理任务都按“异步作业”处理：

1. 前端调用某个 `POST` 接口创建任务
2. 后端立即返回一个 `job_id`
3. 前端轮询 `GET /api/jobs/{job_id}` 查看状态
4. 前端按需调用 `GET /api/jobs/{job_id}/log` 获取日志尾部内容

任务状态一般有这些值：

- `queued`: 已创建，尚未启动
- `running`: 正在运行
- `completed`: 成功完成
- `failed`: 执行失败
- `stopped`: 被手动停止

任务元数据会持久化到本地 SQLite：

- 数据库位置：`fastapi_backend/state/jobs.db`
- FastAPI 重启后，历史任务仍可通过 `job_id` 查询
- 日志、模型输出、`results.csv` 仍然保存在原有文件路径，不写入数据库
- 重启恢复时，如果旧任务进程还活着，会继续显示为 `running`
- 如果旧任务进程已经不存在，后端会结合 `results.csv` 是否存在来把状态修正为 `completed` 或 `failed`

## 通用返回结构

大部分创建任务接口和任务详情接口返回结构一致，核心字段如下：

```json
{
  "job_id": "0d9d7b8d0f9b4b2fa6f9470f4aaf1234",
  "job_type": "classical",
  "script_name": "train.sh",
  "status": "running",
  "command": [
    "bash",
    "/home/zjr123/LLM4CVD-main/scripts/train.sh",
    "reveal",
    "ReGVD",
    "0-512",
    "0"
  ],
  "pid": 12345,
  "created_at": "2026-04-05T08:00:00.000000",
  "started_at": "2026-04-05T08:00:00.100000",
  "finished_at": null,
  "return_code": null,
  "output_dir": "/home/zjr123/LLM4CVD-main/outputs/ReGVD/reveal_0-512",
  "log_file": "/home/zjr123/LLM4CVD-main/outputs/ReGVD/reveal_0-512/train_ReGVD_reveal_0-512.log",
  "result_csv": "/home/zjr123/LLM4CVD-main/outputs/ReGVD/reveal_0-512/results.csv",
  "params": {
    "action": "train",
    "dataset_name": "reveal",
    "model_name": "ReGVD",
    "length": "0-512",
    "cuda": "0"
  },
  "error_message": null
}
```

字段说明：

- `job_id`: 任务唯一 ID
- `job_type`: 任务分类
- `script_name`: 实际调用的脚本
- `status`: 当前状态
- `command`: 最终执行的命令
- `pid`: 任务进程号
- `output_dir`: 输出目录
- `log_file`: 日志文件路径
- `result_csv`: 结果 CSV 路径
- `params`: 创建任务时提交的参数
- `error_message`: 启动失败时的错误信息

## 接口总览

### 1. `GET /health`

用途：检查服务是否存活。

响应示例：

```json
{
  "status": "ok"
}
```

### 2. `GET /api/meta/options`

用途：给前端表单提供候选项，例如模型名、数据集目录、长度区间、类别不平衡比例。

返回内容包括：

- `graph_models`
- `llm_models`
- `datasets.regular`
- `datasets.imbalance`
- `regular_lengths`
- `imbalance_lengths`
- `imbalance_ratios`

请求示例：

```bash
curl http://127.0.0.1:8000/api/meta/options
```

### 2.1 `GET /api/outputs`

用途：浏览 `outputs/` 目录下的文件和子目录，适合前端做文件树或结果面板。

查询参数：

- `relative_path`: 相对于 `outputs/` 的路径，默认空字符串，表示根目录

调用示例：

```bash
curl "http://127.0.0.1:8000/api/outputs"
curl "http://127.0.0.1:8000/api/outputs?relative_path=UniXcoder_imbalance"
```

返回示例：

```json
{
  "base_dir": "/home/zjr123/LLM4CVD-main/outputs",
  "relative_path": "UniXcoder_imbalance",
  "entries": [
    {
      "name": "bigvul_cwe119_1_1",
      "relative_path": "UniXcoder_imbalance/bigvul_cwe119_1_1",
      "entry_type": "directory",
      "size": null
    }
  ]
}
```

### 2.2 `GET /api/outputs/text`

用途：读取 `outputs/` 中的文本文件内容，适合前端显示 `.log`、`.csv`、`.py` 等文本。

查询参数：

- `relative_path`: 相对于 `outputs/` 的文件路径
- `max_chars`: 最多返回多少字符，默认 `200000`

调用示例：

```bash
curl "http://127.0.0.1:8000/api/outputs/text?relative_path=UniXcoder_imbalance/bigvul_cwe119_1_1/results.csv"
```

### 2.3 `GET /api/outputs/file`

用途：直接返回 `outputs/` 中的文件，适合前端加载图片、下载 CSV、打开日志原文件。

查询参数：

- `relative_path`: 相对于 `outputs/` 的文件路径

调用示例：

```bash
curl -O "http://127.0.0.1:8000/api/outputs/file?relative_path=UniXcoder_imbalance/bigvul_cwe119_1_1/results.csv"
```

图片场景下，前端可以直接把这个接口当作 `img` 的 `src`：

```html
<img src="http://127.0.0.1:8000/api/outputs/file?relative_path=UniXcoder_imbalance/bigvul_cwe119_1_1/true_distribution.png" />
```

### 3. `POST /api/jobs/classical`

用途：创建传统模型训练或测试任务，对应：

- `scripts/train.sh`
- `scripts/test.sh`

请求体：

```json
{
  "action": "train",
  "dataset_name": "reveal",
  "model_name": "ReGVD",
  "length": "0-512",
  "cuda": "0"
}
```

字段说明：

- `action`: `train` 或 `test`
- `dataset_name`: 数据集名
- `model_name`: `Devign`、`ReGVD`、`GraphCodeBERT`、`CodeBERT`、`UniXcoder`
- `length`: 数据长度区间，例如 `0-512`
- `cuda`: 使用的 GPU 编号，默认 `0`

调用示例：

```bash
curl -X POST http://127.0.0.1:8000/api/jobs/classical \
  -H "Content-Type: application/json" \
  -d '{
    "action": "train",
    "dataset_name": "reveal",
    "model_name": "ReGVD",
    "length": "0-512",
    "cuda": "0"
  }'
```

### 4. `POST /api/jobs/classical-imbalance`

用途：创建传统模型的不平衡数据训练或测试任务，对应：

- `scripts/train_imbalance.sh`
- `scripts/test_imbalance.sh`

请求体：

```json
{
  "action": "train",
  "dataset_name": "draper",
  "model_name": "CodeBERT",
  "length": "1",
  "pos_ratio": "0.2",
  "cuda": "0"
}
```

字段说明：

- `action`: `train` 或 `test`
- `dataset_name`: 数据集名
- `model_name`: 传统模型名
- `length`: 长度区间
- `pos_ratio`: 正样本比例
- `cuda`: GPU 编号

### 5. `POST /api/jobs/llm`

用途：创建大模型常规微调或推理任务，对应：

- `scripts/finetune.sh`
- `scripts/inference.sh`

请求体：

```json
{
  "action": "finetune",
  "dataset_name": "reveal",
  "model_name": "llama3.1",
  "length": "0-512",
  "batch_size": 16,
  "cuda": "0"
}
```

字段说明：

- `action`: `finetune` 或 `inference`
- `dataset_name`: 数据集名
- `model_name`: `llama2`、`llama3`、`llama3.1`、`codellama`
- `length`: 长度区间
- `batch_size`: 仅 `finetune` 时必填
- `cuda`: GPU 编号

如果 `action=inference`，可以不传 `batch_size`。

### 6. `POST /api/jobs/llm-imbalance`

用途：创建大模型在不平衡数据上的微调或推理任务，对应：

- `scripts/finetune_imbalance.sh`
- `scripts/inference_imbalance.sh`

请求体：

```json
{
  "action": "finetune",
  "dataset_name": "draper",
  "model_name": "llama3.1",
  "pos_ratio": "0.2",
  "cuda": "0"
}
```

### 7. `POST /api/jobs/ablation`

用途：创建 LoRA 消融实验任务，对应：

- `scripts/finetune_ablation.sh`
- `scripts/inference_ablation.sh`

请求体：

```json
{
  "action": "finetune",
  "dataset_name": "reveal",
  "model_name": "llama3.1",
  "r": 8,
  "alpha": 16,
  "cuda": "0"
}
```

字段说明：

- `r`: LoRA rank
- `alpha`: LoRA alpha

### 8. `POST /api/jobs/to-graph`

用途：执行图转换任务，对应：

- `scripts/to_graph.sh`

请求体：

```json
{
  "dataset_name": "reveal",
  "length": "0-512"
}
```

调用示例：

```bash
curl -X POST http://127.0.0.1:8000/api/jobs/to-graph \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_name": "reveal",
    "length": "0-512"
  }'
```

### 9. `GET /api/jobs`

用途：获取当前后端已持久化的全部任务列表。

说明：

- 返回值是数组
- 按任务创建时间倒序排列
- 包含服务重启前已写入 SQLite 的历史任务

调用示例：

```bash
curl http://127.0.0.1:8000/api/jobs
```

### 10. `GET /api/jobs/{job_id}`

用途：查询某个任务的详细状态。

调用示例：

```bash
curl http://127.0.0.1:8000/api/jobs/0d9d7b8d0f9b4b2fa6f9470f4aaf1234
```

前端通常用这个接口做轮询。

### 11. `GET /api/jobs/{job_id}/log`

用途：读取某个任务日志文件的最后若干行，适合前端实时展示日志面板。

查询参数：

- `lines`: 返回最后多少行，默认 `200`，范围 `1` 到 `2000`

调用示例：

```bash
curl "http://127.0.0.1:8000/api/jobs/0d9d7b8d0f9b4b2fa6f9470f4aaf1234/log?lines=100"
```

返回示例：

```json
{
  "job_id": "0d9d7b8d0f9b4b2fa6f9470f4aaf1234",
  "log_file": "/home/zjr123/LLM4CVD-main/outputs/ReGVD/reveal_0-512/train_ReGVD_reveal_0-512.log",
  "content": "epoch 1 ...\nepoch 2 ..."
}
```

### 12. `POST /api/jobs/{job_id}/stop`

用途：终止正在运行的任务。

说明：

- 后端会尝试结束对应进程组
- 成功后任务状态会变成 `stopped`

调用示例：

```bash
curl -X POST http://127.0.0.1:8000/api/jobs/0d9d7b8d0f9b4b2fa6f9470f4aaf1234/stop
```

## 前端对接建议

推荐的基本流程：

1. 页面初始化时调用 `GET /api/meta/options`
2. 页面初始化时也可以调用 `GET /api/outputs` 加载已有输出目录
3. 用户选择参数后，调用对应的 `POST /api/jobs/...`
4. 拿到 `job_id` 后，每隔 2 到 5 秒轮询 `GET /api/jobs/{job_id}`
5. 如果需要实时日志，同时轮询 `GET /api/jobs/{job_id}/log?lines=100`
6. 任务完成后，用 `output_dir` 对应的相对路径继续调用 `GET /api/outputs`
7. 文本文件用 `GET /api/outputs/text`，图片和下载文件用 `GET /api/outputs/file`
8. 当任务状态变为 `completed`、`failed` 或 `stopped` 时停止轮询

## 一个完整示例

下面以传统模型训练为例：

### 第一步：提交训练任务

```bash
curl -X POST http://127.0.0.1:8000/api/jobs/classical \
  -H "Content-Type: application/json" \
  -d '{
    "action": "train",
    "dataset_name": "reveal",
    "model_name": "ReGVD",
    "length": "0-512",
    "cuda": "0"
  }'
```

### 第二步：查询任务状态

```bash
curl http://127.0.0.1:8000/api/jobs/<job_id>
```

### 第三步：查看日志

```bash
curl "http://127.0.0.1:8000/api/jobs/<job_id>/log?lines=100"
```

### 第四步：任务结束后读取输出

返回的任务信息里会提供：

- `output_dir`
- `log_file`
- `result_csv`

前端可以把这些路径显示出来，或者交给后续文件下载接口继续扩展。

## 当前限制

- 运行中任务的实时控制仍依赖当前进程持有的子进程信息；服务重启后虽然可以恢复任务记录，但无法恢复原始 Python `Popen` 对象
- 重启恢复时，对旧任务是否完成的判断主要依赖进程是否仍存在，以及 `results.csv` 是否存在，不能完全替代更严格的任务队列系统
- 日志接口只返回日志尾部文本
- 输出文件下载通过 `GET /api/outputs/file` 提供，数据库只保存文件路径元数据

## 后续可扩展方向

- 增加按任务类型筛选列表
- 增加 WebSocket 日志推送
- 增加鉴权和用户隔离
