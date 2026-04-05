`LLM_TEST` 下新增了 3 个独立脚本，参数默认从 `LLM_TEST/.env` 和 `exp.yaml` 读取。

推荐做法：

```bash
cp LLM_TEST/.env.example LLM_TEST/.env
```

然后只修改 `LLM_TEST/.env`，`exp.yaml` 里使用环境变量占位符，不再直接写死参数或密钥。

默认直接运行：

```bash
python LLM_TEST/extract_positive_samples.py
python LLM_TEST/llm_api_judge.py
python LLM_TEST/recompute_metrics.py
```

如果想换配置文件：

```bash
python LLM_TEST/extract_positive_samples.py --config LLM_TEST/exp.yaml --env_file LLM_TEST/.env
python LLM_TEST/llm_api_judge.py --config LLM_TEST/exp.yaml --env_file LLM_TEST/.env
python LLM_TEST/recompute_metrics.py --config LLM_TEST/exp.yaml --env_file LLM_TEST/.env
```

命令行参数仍然可以临时覆盖 `exp.yaml` 中的值，例如：

```bash
python LLM_TEST/llm_api_judge.py --model your_model_name --api_key your_key
```

配置来源：

- `LLM_TEST/.env`：真正的可变参数和密钥
- `LLM_TEST/exp.yaml`：参数模板，使用 `${VAR_NAME}` 引用环境变量

`exp.yaml` 主要分成 4 段：

- `common`：公共目录参数
- `extract`：阳性样例提取参数
- `llm`：API 模型、提示词、重试等参数
- `metrics`：合并和重算指标参数

默认输出目录：

- 中间样例：`LLM_TEST/intermediate/<dataset_id>/`
- 大模型输出：`LLM_TEST/output/<dataset_id>/`

主要文件：

- `positive_indices.json`：阳性样例索引
- `positive_samples.json`：阳性样例及其原始代码
- `llm_predictions.csv`：大模型判断结果
- `merged_results.csv`：合并后的最终预测
- `metrics.json`：重算后的指标

说明：

- `API_KEY_ENV=OPENAI_API_KEY` 表示脚本会去 `.env` 中读取 `OPENAI_API_KEY`
- 如果某个 `${VAR_NAME}` 在 `.env` 中缺失，脚本会直接报错，提示缺哪个变量
