# ChemEvalBench

A modular, production-ready benchmarking suite for evaluating Large Language Models (LLMs) and Multimodal Vision Language Models (MLLMs) on the **[ChemEval](https://huggingface.co/datasets/Ooo1/ChemEval)** benchmark.

ChemEvalBench provides unified model interfaces, cost-effective asynchronous batch APIs, robust output extraction (handling markdown fences, reasoning blocks, and JSON outputs), deterministic code-based grading, and cached LLM-as-a-judge scoring across chemistry domains:
- **Organic & Inorganic Chemistry**: Nomenclature (SMILES, SELFIES, IUPAC, molecular formulas), property prediction, reaction type recognition, reaction yield.
- **Synthesis Planning**: Single-step & multi-step retrosynthesis, reaction conditions (catalysts, ligands, solvents, temperature, time), reaction intermediates.
- **Chemical Literature**: Topic classification, entity extraction, relation extraction, abstract and outline generation.
- **Spectroscopy & Multimodal Vision**: Infrared (IR), Raman, UV-Vis, NMR, Mass Spectrometry (MS), XRD, 2D chemical structures, potential energy surfaces, and kinetic phase diagrams.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Installation & Environment Setup](#installation--environment-setup)
- [Repository Structure](#repository-structure)
- [Running Predictions (`create.py`)](#running-predictions-createpy)
  - [1. API Backends via LiteLLM (Direct / Synchronous)](#1-api-backends-via-litellm-direct--synchronous)
  - [2. Local Open-Weight Models via vLLM (`hf` Backend)](#2-local-open-weight-models-via-vllm-hf-backend)
  - [3. Asynchronous Batch APIs (OpenAI & Anthropic 50% Off)](#3-asynchronous-batch-apis-openai--anthropic-50-off)
- [Running Evaluations (`evaluate.py`)](#running-evaluations-evaluatepy)
  - [Free Code-Graded Evaluation (No LLM Judge)](#free-code-graded-evaluation-no-llm-judge)
  - [Full Evaluation with LLM-as-a-Judge](#full-evaluation-with-llm-as-a-judge)
  - [Pre-flight Inspection Mode](#pre-flight-inspection-mode)
  - [Evaluating Multiple Runs (Mean ± Std)](#evaluating-multiple-runs-mean--std)
- [Cost Estimation & Budgeting (`cost_estimate.py`)](#cost-estimation--budgeting-cost_estimatepy)
- [Important Options & Gotchas](#important-options--gotchas)

---

## Quick Start

```bash
# 1. Activate your virtual environment and configure keys
source llm_bench_env/bin/activate
cp .env.example .env   # fill in your OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.

# 2. Run a smoke test on the text benchmark (e.g. 5 samples with GPT-4o)
python text_eval/create.py --backend litellm --model gpt-4o --name gpt4osmoke --limit 5

# 3. Evaluate the generated predictions without calling an LLM judge
python text_eval/evaluate.py --inputs dataset_1times/gpt4osmoke/0shot/gpt4osmoke_1t.json --skip-judge --out results/gpt4osmoke_eval
```

---

## Installation & Environment Setup

### 1. Python Environment
Python 3.10+ is recommended. If you already have an environment created:
```bash
source llm_bench_env/bin/activate
```
Or create a fresh virtual environment:
```bash
python3 -m venv llm_bench_env
source llm_bench_env/bin/activate
```

### 2. Dependencies
Install the required scientific, chemical, and evaluation packages:
```bash
pip install -r requirements.txt
```

Additional runtime packages for LLM inference, datasets, and tracing:
```bash
pip install datasets litellm tiktoken pillow
```
For local GPU inference with vLLM:
```bash
pip install vllm transformers
```
For batch API clients and Langfuse tracing (optional):
```bash
pip install openai anthropic langfuse
```

### 3. Environment Variables
Copy `.env.example` to `.env` and configure your API credentials:
```bash
cp .env.example .env
```
Key configuration entries:
```env
# API Keys
OPENAI_API_KEY="sk-..."
ANTHROPIC_API_KEY="sk-ant-..."
GEMINI_API_KEY="..."

# Optional: Langfuse tracing for cost & token observability
LANGFUSE_PUBLIC_KEY="pk-lf-..."
LANGFUSE_SECRET_KEY="sk-lf-..."
LANGFUSE_HOST="https://cloud.langfuse.com"
```

---

## Repository Structure

```
ChemEvalBench/
├── text_eval/                     # Textual ChemEval evaluation pipeline
│   ├── create.py                  # Model inference on dataset split 'text'
│   ├── evaluate.py                # Scoring script (code-graded + LLM judge)
│   ├── model.py                   # LiteLLM, vLLM, and Batch API model wrappers
│   ├── cost_estimate.py           # Token & cost projection tool
│   └── repo_constants.json        # Official LLM-as-a-judge scoring prompt templates
├── multimodal_eval/               # Multimodal ChemEval evaluation pipeline
│   ├── create.py                  # Vision model inference on split 'multimodal'
│   ├── evaluate.py                # Multimodal scoring script
│   ├── model.py                   # Multimodal LiteLLM, vLLM, & Batch wrappers
│   └── cost_estimate.py           # Vision token & cost projection tool
├── requirements.txt               # Base chemistry & evaluation dependencies
├── .env.example                   # Template for API keys and observability
└── README.md
```

---

## Running Predictions (`create.py`)

Both `text_eval/create.py` and `multimodal_eval/create.py` support crash resumption (if interrupted, re-running the same command resumes from the last completed row).

> [!IMPORTANT]
> **Naming Rule**: The `--name` parameter is used for output paths. It **must not contain underscores** (`_`) and **must not end with the letter `t`** (e.g., use `gpt4o`, `qwen7b`, `sonnet`, not `gpt_4o` or `test_t`).

---

### 1. API Backends via LiteLLM (Direct / Synchronous)

LiteLLM supports OpenAI, Anthropic, Google Gemini, DeepSeek, Mistral, Ollama, and any OpenAI-compatible server.

#### Text Benchmark
```bash
# OpenAI GPT-4o
python text_eval/create.py --backend litellm --model gpt-4o --name gpt4o

# Anthropic Claude 3.5 Sonnet
python text_eval/create.py --backend litellm --model claude-3-5-sonnet-20241022 --name sonnet

# Google Gemini 1.5 Pro
python text_eval/create.py --backend litellm --model gemini/gemini-1.5-pro --name gemini15pro

# Smoke test (e.g. first 10 rows)
python text_eval/create.py --backend litellm --model gpt-4o-mini --name gpt4omini --limit 10
```

#### Multimodal Benchmark
```bash
# OpenAI GPT-4o
python multimodal_eval/create.py --backend litellm --model gpt-4o --name gpt4omm

# Claude 3.5 Sonnet (Vision)
python multimodal_eval/create.py --backend litellm --model claude-3-5-sonnet-20241022 --name sonnetmm

# Smoke test on multimodal
python multimodal_eval/create.py --backend litellm --model gpt-4o-mini --name minimm --limit 5
```

---

### 2. Local Open-Weight Models via vLLM (`hf` Backend)

For running open-source models locally on your GPU(s) with high throughput using `vllm`.

#### Text Benchmark
```bash
# Meta LLaMA 3.1 8B Instruct
python text_eval/create.py \
  --backend hf \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --name llama8b \
  --batch-size 32 \
  --max-model-len 4096

# Qwen 2.5 7B Instruct
python text_eval/create.py \
  --backend hf \
  --model Qwen/Qwen2.5-7B-Instruct \
  --name qwen7b \
  --batch-size 32

# Local weights from disk path
python text_eval/create.py \
  --backend hf \
  --model /path/to/local/model-weights \
  --name localmodel \
  --batch-size 16
```

#### Multimodal Benchmark
```bash
# Qwen2-VL 7B Instruct
python multimodal_eval/create.py \
  --backend hf \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --name qwen2vl \
  --batch-size 16 \
  --max-model-len 4096

# InternVL2 / LLaVA models
python multimodal_eval/create.py \
  --backend hf \
  --model OpenGVLab/InternVL2-8B \
  --name internvl8b \
  --batch-size 8
```

---

### 3. Asynchronous Batch APIs (OpenAI & Anthropic 50% Off)

For evaluating large models with significant discounts (typically 50% cheaper) without hitting aggressive per-minute rate limits:

Supported backends: `openai-batch`, `anthropic-batch`.

#### Step 1: Submit the batch job
```bash
# Submit OpenAI text batch
python text_eval/create.py \
  --backend openai-batch \
  --model gpt-4o \
  --name gpt4obatch \
  --batch-action submit

# Submit Anthropic multimodal batch
python multimodal_eval/create.py \
  --backend anthropic-batch \
  --model claude-3-5-sonnet-20241022 \
  --name sonnetbatch \
  --batch-action submit
```
*A state file tracking the batch ID and progress will be created in the output directory.*

#### Step 2: Check batch status
```bash
# Check status once
python text_eval/create.py \
  --backend openai-batch \
  --name gpt4obatch \
  --batch-action status
```

#### Step 3: Retrieve completed predictions
```bash
# Poll until completion and download output
python text_eval/create.py \
  --backend openai-batch \
  --name gpt4obatch \
  --batch-action retrieve \
  --wait \
  --poll-interval 60
```

---

## Running Evaluations (`evaluate.py`)

The evaluation script routes every task to either deterministic code grading or LLM-as-a-judge scoring according to the benchmark specification:
- **Code-Graded Tasks**: Classification (Accuracy), Regression (NRMSE, RMSE, RAE, $R^2$), Molecular conversion (Tanimoto similarity, BLEU, Levenshtein edit distance), Reagent & Catalyst recommendation (Precision, Recall, F1).
- **LLM-Judged Tasks**: Fill-in-the-blank, Multiple-choice, Short answer, Calculations, Synthesis pathways, Reaction intermediates, Paper abstract & outline generation, Spectral descriptions.

---

### Free Code-Graded Evaluation (No LLM Judge)

To score all code-based tasks immediately without incurring any LLM API costs, add `--skip-judge`:

#### Text Evaluation
```bash
python text_eval/evaluate.py \
  --inputs dataset_1times/gpt4o/0shot/gpt4o_1t.json \
  --skip-judge \
  --out results/gpt4o_text_eval
```

#### Multimodal Evaluation
```bash
python multimodal_eval/evaluate.py \
  --inputs results/gpt4omm/gpt4omm_multimodal.jsonl \
  --skip-judge \
  --out results/gpt4omm_eval
```

---

### Full Evaluation with LLM-as-a-Judge

LLM-judged tasks use the official prompt templates in `repo_constants.json`. Responses are automatically cached in `judge_cache.jsonl` within the output folder so you never pay twice for the same judgment.

```bash
python text_eval/evaluate.py \
  --inputs dataset_1times/gpt4o/0shot/gpt4o_1t.json \
  --judge-model gpt-4o-2024-08-06 \
  --judge-workers 8 \
  --out results/gpt4o_full_eval
```

#### Deterministic Extractor for Multiple Choice & True/False
By default, an LLM judge extracts the option choice. You can toggle deterministic rule-based regex extraction:
```bash
python text_eval/evaluate.py \
  --inputs dataset_1times/gpt4o/0shot/gpt4o_1t.json \
  --mc-tf-extractor rule \
  --out results/gpt4o_full_eval
```

#### Dump Prompts Without Calling the Judge
To inspect judge prompts or run them offline:
```bash
python text_eval/evaluate.py \
  --inputs dataset_1times/gpt4o/0shot/gpt4o_1t.json \
  --dump-judge-prompts \
  --out results/gpt4o_prompts
```

---

### Pre-flight Inspection Mode

Verify that model responses are properly parsed and all files are routed before running a full evaluation:
```bash
python text_eval/evaluate.py \
  --inputs dataset_1times/gpt4o/0shot/gpt4o_1t.json \
  --inspect
```
This prints the number of rows, JSON extraction success rate, task categories, and paper task IDs.

---

### Evaluating Multiple Runs (Mean ± Std)

To match the paper's multi-run reporting (averaging across multiple runs):
```bash
python text_eval/evaluate.py \
  --inputs \
    dataset_1times/gpt4o_run1/0shot/gpt4o_run1_1t.json \
    dataset_1times/gpt4o_run2/0shot/gpt4o_run2_1t.json \
    dataset_1times/gpt4o_run3/0shot/gpt4o_run3_1t.json \
  --judge-model gpt-4o-2024-08-06 \
  --out results/gpt4o_3runs
```
Outputs:
- `summary.csv`: Machine-readable results with mean and standard deviation per task.
- `summary.json`: Detailed hierarchical dictionary of results.
- `per_file.json`: Granular breakdown per dataset source file.

---

## Cost Estimation & Budgeting (`cost_estimate.py`)

ChemEval tasks have high reasoning and prompt token counts. The cost estimator projects costs before you run benchmarks, accounting for hidden reasoning/thinking tokens (e.g. OpenAI o-series, Claude reasoning effort):

```bash
# Text cost estimation (default models)
python text_eval/cost_estimate.py --by-task

# Estimate specific models with batch pricing
python text_eval/cost_estimate.py --models gpt4o claude-haiku-5-5 --batch

# Multimodal cost estimation (with image tile breakdown)
python multimodal_eval/cost_estimate.py --by-task

# Calibrate estimates using a real Langfuse CSV export
python text_eval/cost_estimate.py --calibration-csv langfuse_export.csv --by-task
```

---

## Important Options & Gotchas

1. **Output Directory Conventions**:
   - `text_eval/create.py` outputs to `dataset_1times/{name}/0shot/{name}_1t.json`.
   - `multimodal_eval/create.py` outputs to `results/{name}/{name}_multimodal.jsonl`.
2. **Resumption**:
   - If an inference run is cancelled or crashes, re-run the exact same command. The script checks already-completed rows and resumes seamlessly.
3. **0-shot vs 3-shot Detection**:
   - `text_eval/evaluate.py` automatically detects shot configuration from record fields or file paths. You can restrict evaluation via `--shots 0` or `--shots 0 3`.
4. **Judge Cache**:
   - Judge outputs are saved to `{out_dir}/judge_cache.jsonl`. If you re-run evaluation with modified parameters or need to re-score a run, existing judgments are reused instantly.
5. **vLLM Context Length**:
   - When serving local models with `--backend hf`, pass `--max-model-len 4096` or higher if your model requires longer chemistry prompts.
