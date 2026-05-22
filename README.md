
# MOSS🍀: Multimodal OCR for Structured Markup Sequencing

A multimodal OCR model for **block-level** layout understanding. It emits structured markup—math as LaTeX, tables as HTML, and body text as Markdown—with support for Japanese, Chinese, and English.

---

github: https://github.com/patsnap/Hiro-MOSS-OCR


## Overview

| Component | Details |
|-----------|---------|
| Training | Trained from scratch; any-resolution images; training data on the order of **~50M** samples |
| Encoder (~90M) | NaViT with 2D RoPE |
| Connector (~13.5M) | SwiGLU with patch merger |
| Decoder (~216.6M) | Transformer (pre-norm, RoPE, GQA, SwiGLU) |
| **Total parameters** | **~320.8M** |

**Tasks and output formats**

| Task | Output |
|------|--------|
| `math` | LaTeX |
| `table` | HTML |
| `text` | Markdown |

**Languages:** JP, CN, EN.

---

## Benchmarks

### OmniDocBench v1.5 (with ground-truth layout labels)

| Model               | Param | Table(TEDS)  |  Math(CDM) | Text (Edit Similarity) | Overall |
|--------------------|-------|-------|------------|--------------------------|---------|
| dolphin            | 0.3B  | 77.08 | 93.88      | 90.96                    | 87.31   |
| Monkey OCR Pro 1.2B| 1.2B  | 83.89 | 94.31      | 93.07                    | 90.42   |
| Mineru 2.5         | 1.2B  | 87.90 | 95.94      | 93.25                    | 92.36   |
| Mineru 2.5 Pro     | 1.2B  | 92.46 | 97.24      | 93.98                    | 94.56   |
| Paddle VL          | 0.9B  | 90.57 | 96.87      | 94.34                    | 93.93   |
| Paddle VL 1.5      | 0.9B  | 90.79 | 97.28      | 94.56                    | 94.21   |
| GLM-OCR            | 0.9B  | 93.71 | 97.74      | 96.44                    | 95.96   |
| MOSS-OCR-0.3B      | 0.3B  | 90.33 | 95.56      | 95.01                    | 93.63   |

### In-house Benchmark (Patent Domain)

| Model               | Param | Table (TEDS)  | CDM (Math) | Overall |
|--------------------|-------|-------|------------|---------|
| dolphin            | 0.3B  | 75.97 | 94.36      | 85.17   |
| Monkey OCR Pro 1.2B| 1.2B  | 78.39 | 93.01      | 85.70   |
| Mineru 2.5         | 1.2B  | 84.27 | 95.28      | 89.78   |
| Mineru 2.5 Pro     | 1.2B  | 87.97 | 96.56      | 92.27   |
| Paddle VL          | 0.9B  | 85.27 | 94.85      | 90.06   |
| Paddle VL 1.5      | 0.9B  | 81.76 | 94.72      | 88.24   |
| GLM-OCR            | 0.9B  | 86.58 | 96.07      | 91.33   |
| MOSS-OCR-0.3B      | 0.3B  | 91.64 | 95.34      | 93.49   |


###  Inference Speed on a Single RTX 4090 (vLLM)

| Model               | Param | QPS (it/s)  |
|--------------------|-------|--------------|
| Mineru 2.5         | 1.2B  | 29.49     | 
| MOSS-OCR-0.3B      | 0.3B  | 58.77    | 


---

## Requirements

- **Python** ≥ 3.12 ([uv](https://github.com/astral-sh/uv) recommended)
- **CUDA Graph / Transformers path:** GPU with PyTorch CUDA; see `pyproject.toml` for dependencies
- **vLLM serving:** GPU required; after install, run the bundled patch script so vLLM registers the MOSS model (below)

---

## Model weights

| Model | Download | Precision |
|-------|----------|-----------|
| Hiro-MOSS-OCR-0.3B | [Hugging Face — PatSnap/Hiro-MOSS-OCR-0.3B](https://huggingface.co/PatSnap/Hiro-MOSS-OCR-0.3B) | FP32 / BF16 |

Download the checkpoint to a local directory and point `MODEL_PATH` to it in the commands below.

---

## Install (from source)

```bash
git clone https://github.com/patsnap/Hiro-MOSS-OCR
cd Hiro-MOSS-OCR

uv python pin 3.12
uv venv .venv
source .venv/bin/activate
uv sync

# Copy MOSS patches into the installed vLLM package (run after uv sync)
bash scripts/vllm_adapter.sh
```

`scripts/vllm_adapter.sh` copies files from `moss_ocr/static/vllm_patches/` (matching the pinned vLLM version) into the `vllm` package under your environment so `vllm serve` can load MOSS.

---

## Usage

### 1. CUDA Graph + Transformers (local inference)

Single-GPU local inference via Transformers and the CUDA Graph path.

```python
from moss_ocr.inferer.cuda_graph import MOSSv1d6Runner

model_path = "/path/to/Hiro-MOSS-OCR-0.3B"
runner = MOSSv1d6Runner(model_path=model_path)

img_path = "/path/to/your/image.png"
task = "text"  # "table" | "math" | "text"

result = runner.run(img=img_path, task=task)
print(result)
```

CLI (same as the bundled example):

```bash
uv run python moss_ocr/examples/run_with_cuda_graph.py \
  --model_path /path/to/Hiro-MOSS-OCR-0.3B \
  --task text \
  --img_path /path/to/your/image.png
```

---

### 2. vLLM server + OpenAI-compatible client

**Step 1 — start vLLM**

Set `MODEL_PATH` to your local Hugging Face checkout:

```bash
export MODEL_PATH=/path/to/Hiro-MOSS-OCR-0.3B

uv run vllm serve "$MODEL_PATH" \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 16384 \
  --port 8088 \
  --served-model-name moss-v1d6-0.3b
```

**Step 2 — client**

`MOSSOCRv1d6vLLMRunner` expects an OpenAI-compatible `base_url` (include the `/v1` suffix):

```python
from moss_ocr.inferer.vllm import MOSSOCRv1d6vLLMRunner

runner = MOSSOCRv1d6vLLMRunner(url="http://127.0.0.1:8088/v1")

img_path = "/path/to/your/image.png"
task = "text"  # "table" | "math" | "text"

result = runner.run(img=img_path, task=task)
print(result)
```

CLI:

```bash
uv run python moss_ocr/examples/run_with_vllm.py \
  --url http://127.0.0.1:8088/v1 \
  --task text \
  --img_path /path/to/your/image.png
```

The default `--served-model-name` must match the client’s `MODEL_NAME` (`moss-v1d6-0.3b`). If you change `--served-model-name`, pass `model_path=<your served name>` when constructing `MOSSOCRv1d6vLLMRunner` (the base class uses this as the API `model` field).

---

## Acknowledgements

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [MinerU](https://github.com/opendatalab/MinerU)
- [GLM-OCR](https://github.com/zai-org/GLM-OCR)
- [Dolphin](https://github.com/bytedance/Dolphin)
- [Monkey-OCR](https://github.com/yuliang-liu/MonkeyOCR)
- [Smol-Docling](https://huggingface.co/docling-project/SmolDocling-256M-preview)
