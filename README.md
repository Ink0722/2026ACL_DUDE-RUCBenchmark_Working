# Don't Click That: Teaching Web Agents to Resist Deceptive Interfaces
<p align="center">
  <a href="TODO"><img src="https://img.shields.io/badge/Paper-ACL%202026-blue" height="23"></a>
  <a href="https://huggingface.co/datasets/Ink0722/Real-UI-Clickboxes"><img src="https://img.shields.io/badge/Data-Real--UI--Clickboxes-green" height="23"></a>
</p>

Codebase for the ACL 2026 submission on Don't Click That: Teaching Web Agents to Resist Deceptive Interfaces. This repository focuses on a full pipeline for training and evaluating web-browsing click judges under deceptive UI conditions:

- Stage 1: evaluator training
- Stage 2: experience optimization (recommended; requires ZHIPUAI_API_KEY)
- Inference: run the agent with an evaluator in the loop

This repository does not ship pretrained checkpoints. Users are expected to run Stage 1 (and optionally Stage 2) locally.

Pipeline: Stage 1 → Stage 2 → Inference

## 🚀 Quickstart (Stage 1 → Stage 2 → Inference)

### 0) 🧱 Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1) 📦 Download dataset (from Hugging Face)

Dataset source:
- https://huggingface.co/datasets/Ink0722/Real-UI-Clickboxes

Recommended command:

```bash
python data/download.py
```

If your annotation filename differs from the default expected by your local setup, update `DATA_PATH` in `.env`.

### 2) 🧩 Configure environment variables

This repository uses centralized configuration via environment variables and `src/config.py`.

Create a local `.env` file to override defaults:

```env
DATASET_ROOT=data/Real-UI-Clickboxes
DATA_PATH=data/Real-UI-Clickboxes/annotations.json
IMAGES_DIR=data/Real-UI-Clickboxes/images

STAGE1_ROOT=data/stage1
STAGE2_ROOT=data/stage2
INFERENCE_ROOT=data/inference

DEFAULT_AGENT_MODEL=Qwen/Qwen3-VL-4B-Instruct
DEFAULT_EVAL_MODEL=Qwen/Qwen3-VL-2B-Thinking
DEFAULT_DEVICE=cuda
HF_ENDPOINT=https://hf-mirror.com

ZHIPUAI_API_KEY=your_key_here
```

### 3) ✅ Stage 1: Train the evaluator

```bash
python train/stage1.py
```

Stage 1 writes artifacts under `STAGE1_ROOT`:
- `evaluator_<timestamp>/`
- `stage1_<timestamp>.jsonl`
- `record_<timestamp>.jsonl`

### 4) ✨ Stage 2 (recommended): Experience optimization

Stage 2 improves evaluator behavior via experience summarization. This step is recommended for reproducing the full pipeline.

- If you have `ZHIPUAI_API_KEY` configured, run Stage 2.
- If you do not have `ZHIPUAI_API_KEY`, you can skip this step and proceed to inference.

```bash
python train/stage2.py
```

By default, Stage 2 reads the latest `stage1_*.jsonl` and `evaluator_*` artifacts under `STAGE1_ROOT`, then writes each run to `STAGE2_ROOT/run_<timestamp>/`.

### 5) ▶️ Inference: Run the agent with evaluator

```bash
python agent_runner/run_agent_with_evaluator.py
```

By default, the runner:

- reads `DATASET_ROOT/use_deception.json`, or falls back to `DATA_PATH`
- resolves images relative to `DATASET_ROOT` and `IMAGES_DIR`
- loads the latest `evaluator_*` directory under `STAGE1_ROOT`
- writes inference results to `INFERENCE_ROOT/gui_agent_results_<timestamp>.json`

#### 🗂 Outputs at a glance

| Stage | Purpose | Output location |
|------:|---------|-----------------|
| Stage 1 | Train evaluator | `STAGE1_ROOT/evaluator_<timestamp>/` |
| Stage 2 | Experience optimization | `STAGE2_ROOT/run_<timestamp>/` |
| Inference | Evaluate agent + evaluator loop | `INFERENCE_ROOT/gui_agent_results_<timestamp>.json` |

## 🗺️ Repository Layout

```text
.
|-- agent_runner/
|   |-- llm_agent.py
|   |-- prompt_template.py
|   `-- run_agent_with_evaluator.py   # Inference / evaluation runner
|-- data/
|   `-- download.py                   # Dataset download helper
|-- src/
|   |-- config.py                     # Centralized settings and environment loading
|   |-- model.py                      # Model wrappers and backend builders
|   |-- parser.py                     # Output parsing helpers
|   `-- template.py                   # Evaluation / summarization prompt templates
|-- train/
|   |-- stage1.py                     # Stage 1 evaluator training
|   |-- stage2.py                     # Stage 2 experience optimization
|   |-- datasets.py
|   |-- formatter.py
|   |-- reward.py
|   `-- rule.py
|-- requirements.txt
|-- .gitignore
`-- README.md
```

## 🧰 Environment

Tested with the dependency set in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For GPU usage, install the PyTorch build that matches your CUDA environment before or instead of the pinned `torch` packages if needed.

## 🧩 Configuration

This repository uses centralized configuration via environment variables and `src/config.py`.

1. Create a local `.env` file if you want to override the defaults in `src/config.py`.
2. Fill in the required keys.
3. Adjust paths or default model names if your local setup differs.

Example:

```env
DATASET_ROOT=data/Real-UI-Clickboxes
DATA_PATH=data/Real-UI-Clickboxes/annotations.json  # update if your dataset JSON filename differs
IMAGES_DIR=data/Real-UI-Clickboxes/images

STAGE1_ROOT=data/stage1
STAGE2_ROOT=data/stage2
INFERENCE_ROOT=data/inference

ZHIPUAI_API_KEY=your_key_here
DEFAULT_AGENT_MODEL=Qwen/Qwen3-VL-4B-Instruct
DEFAULT_EVAL_MODEL=Qwen/Qwen3-VL-2B-Thinking
DEFAULT_DEVICE=cuda
HF_ENDPOINT=https://hf-mirror.com
```

### Required Variables

Required for all stages:

- `DATASET_ROOT`: dataset root directory.
- `DATA_PATH`: dataset annotation file path.
- `IMAGES_DIR`: directory for referenced UI screenshots.

Required for Stage 2 only:

- `ZHIPUAI_API_KEY`: required for Stage 2 experience summarization.

### Optional Variables

- `STAGE1_ROOT`
- `STAGE2_ROOT`
- `INFERENCE_ROOT`
- `DEFAULT_AGENT_MODEL`
- `DEFAULT_EVAL_MODEL`
- `DEFAULT_DEVICE`
- `HF_ENDPOINT`

## 📦 Data

The recommended way to obtain the dataset is to use the helper script under `data/`.

Dataset source:
- `https://huggingface.co/datasets/Ink0722/Real-UI-Clickboxes`

Recommended command:

```bash
python data/download.py
```

This downloads the dataset into:

```text
data/
`-- Real-UI-Clickboxes/
```

The downloader uses the Hugging Face dataset repository directly and writes it to a fixed local directory.

After download, the expected dataset layout is:

```text
data/
`-- Real-UI-Clickboxes/
    |-- README.md
    |-- annotations.json or dataset JSON file
    `-- images/
```

If your annotation filename differs from the default expected by your local setup, update `DATA_PATH` in `.env`.

For agent-side inference, `agent_runner/run_agent_with_evaluator.py` first looks for `data/Real-UI-Clickboxes/use_deception.json`. If that file does not exist, it falls back to `DATA_PATH`.

## 🧪 Main Workflows

### 1. Train the evaluator (Stage 1)

```bash
python train/stage1.py
```

By default, Stage 1 reads `DATA_PATH` and `IMAGES_DIR`, then writes artifacts under `STAGE1_ROOT`.

### 2. Run Stage 2 experience optimization (recommended)

```bash
python train/stage2.py
```

By default, Stage 2 reads the latest `stage1_*.jsonl` and `evaluator_*` artifacts under `STAGE1_ROOT`, then writes each run to `STAGE2_ROOT/run_<timestamp>/`.

If `ZHIPUAI_API_KEY` is not available, you can skip this step and directly run the agent with the Stage 1 evaluator.

### 3. Run the agent with evaluator

```bash
python agent_runner/run_agent_with_evaluator.py
```

By default, the runner:

- reads `DATASET_ROOT/use_deception.json`, or falls back to `DATA_PATH`
- resolves images relative to `DATASET_ROOT` and `IMAGES_DIR`
- loads the latest `evaluator_*` directory under `STAGE1_ROOT`
- writes inference results to `INFERENCE_ROOT/gui_agent_results_<timestamp>.json`

## 📊 Output Conventions

- Stage 1 evaluator checkpoints are saved as `STAGE1_ROOT/evaluator_<timestamp>`.
- Stage 1 dataset snapshots are saved as `STAGE1_ROOT/stage1_<timestamp>.jsonl`.
- Stage 1 reward records are saved as `STAGE1_ROOT/record_<timestamp>.jsonl`.
- Stage 2 optimization runs are saved as `STAGE2_ROOT/run_<timestamp>/`.
- Agent runner inference results are saved as `INFERENCE_ROOT/gui_agent_results_<timestamp>.json`.

## 📝 Notes

- The main maintained configuration path is `src/config.py` plus environment variables.
- Stage 1 outputs, Stage 2 outputs, and inference results are intended to live under `data/`.
- Before publishing the repository, rotate any API keys that were ever stored in local code history.

## 📚 Citation

If you use this repository, please cite the associated ACL 2026 paper. Add the final BibTeX entry here after camera-ready details are fixed.

## ⚙️ Runtime Notes

- Local multimodal backends require CUDA GPU. CPU loading is intentionally disabled.
