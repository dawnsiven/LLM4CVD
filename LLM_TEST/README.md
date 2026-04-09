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
python LLM_TEST/llm_local_judge.py
python LLM_TEST/recompute_metrics.py
```

如果你想把“LLM 复测 + 指标重算”合并成一次执行，可以直接运行：

```bash
bash LLM_TEST/run_review.sh
```

默认会读取 `LLM_TEST/exp.yaml` 和 `LLM_TEST/.env`，并复测前 `100` 条样本。

也支持手动覆盖配置文件、环境文件和样本条数：

```bash
bash LLM_TEST/run_review.sh LLM_TEST/exp.yaml LLM_TEST/.env 100
```

如果想换配置文件：

```bash
python LLM_TEST/extract_positive_samples.py --config LLM_TEST/exp.yaml --env_file LLM_TEST/.env
python LLM_TEST/llm_api_judge.py --config LLM_TEST/exp.yaml --env_file LLM_TEST/.env
python LLM_TEST/llm_local_judge.py --config LLM_TEST/exp.yaml --env_file LLM_TEST/.env
python LLM_TEST/recompute_metrics.py --config LLM_TEST/exp.yaml --env_file LLM_TEST/.env
```

命令行参数仍然可以临时覆盖 `exp.yaml` 中的值，例如：

```bash
python LLM_TEST/llm_api_judge.py --model your_model_name --api_key your_key
```

如果想针对某个提示词版本做一轮独立复测，并把结果写到新目录中，可以这样：

```bash
python LLM_TEST/llm_api_judge.py \
  --prompt_file LLM_TEST/Prompt/CWE-119_0.1.txt \
  --limit 100 \
  --output_by_prompt_version

python LLM_TEST/recompute_metrics.py \
  --llm_predictions_csv LLM_TEST/output/<dataset_id>_CWE-119_0.1/llm_predictions.csv \
  --output_dir LLM_TEST/output/<dataset_id>_CWE-119_0.1
```

如果你不想微调，而是想直接用本地 Hugging Face 模型做 zero-shot 判断，可以这样：

```bash
python LLM_TEST/llm_local_judge.py \
  --input_json LLM_TEST/intermediate/<dataset_id>/positive_samples.json \
  --prompt_file LLM_TEST/Prompt/CWE-119.txt \
  --model llama3.2 \
  --output_by_prompt_version
```

说明：

- `--model` 支持别名：`llama2`、`codellama`、`llama3`、`llama3.1`、`llama3.2`
- 也支持直接传本地模型目录或 Hugging Face repo id
- 脚本会优先读取仓库根目录下 `model/` 中已下载的模型
- 输出格式与 `llm_api_judge.py` 保持一致，可直接继续跑 `recompute_metrics.py`

说明：

- `--limit 100`：只测试前 100 条
- `--output_by_prompt_version`：输出到 `LLM_TEST/output/<dataset_id>_<prompt_file_stem>/`
- `--output_name xxx`：如果你想手动指定目录名，也可以直接覆盖自动命名

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
