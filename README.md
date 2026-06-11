# MOSS🍀: Multimodal OCR for Structured Markup Sequencing

[English](README.md) | [简体中文](README_zh.md)

MOSS is a multimodal OCR model **trained from scratch on 50M+ samples** for block-level document understanding. It converts document image regions into structured markup, including LaTeX for formulas, HTML for tables, and Markdown for body text. The model supports Japanese, Chinese, and English.

Repository: [github.com/patsnap/Hiro-MOSS-OCR](https://github.com/patsnap/Hiro-MOSS-OCR)

---

## News and Updates

<details>
<summary>Recent updates</summary>

- **2026-05-28** - CUDA Graph and vLLM inference can now resolve the Hugging Face Hub repo id directly, so `PatSnap/Hiro-MOSS-OCR-0.3B` works without manually downloading the checkpoint first.
- **2026-05-28** - Added a Transformers `AutoModelForCausalLM` quick-call path for smoke tests. This path is convenient but slower than the CUDA Graph and vLLM backends.
- **2026-05-26** - Hiro-MOSS-OCR-0.3B is available on [Hugging Face](https://huggingface.co/PatSnap/Hiro-MOSS-OCR-0.3B).
- **2026-05-26** - The repository includes both local CUDA Graph inference and vLLM serving examples.

</details>

---

## Highlights

- **Trained from scratch on 50M+ samples:** built specifically for structured OCR and document image understanding.
- **Structured outputs:** formula recognition, table reconstruction, and text extraction in task-specific markup formats.
- **Compact model size:** about **320.8M** parameters.
- **Any-resolution image support:** NaViT-style visual encoding with 2D RoPE.
- **Multiple inference paths:** Transformers quick calls, local CUDA Graph inference, and vLLM serving with an OpenAI-compatible client.

---

## Model Overview

| Component | Details |
|-----------|---------|
| Training | Trained from scratch on **50M+** samples with any-resolution images |
| Encoder (~90M) | NaViT with 2D RoPE |
| Connector (~13.5M) | SwiGLU with patch merger |
| Decoder (~216.6M) | Transformer decoder with pre-norm, RoPE, GQA, and SwiGLU |
| **Total parameters** | **~320.8M** |

## Supported Tasks

| Task | Output format |
|------|---------------|
| `math` | LaTeX |
| `table` | HTML |
| `text` | Markdown |

**Languages:** Japanese, Chinese, English.

---

## Related Documents

- [Disclaimer](docs/DISCLAIMER.md) - terms of use, limitations of liability, and data-handling responsibilities.
- [License](LICENSE) - source-code license.

---

## Benchmarks

### OmniDocBench v1.5

Evaluation with ground-truth layout labels.

| Model | Params | Table (TEDS) | Math (CDM) | Text (Edit Similarity) | Overall |
|-------|--------|--------------|------------|-------------------------|---------|
| dolphin | 0.3B | 77.08 | 93.88 | 90.96 | 87.31 |
| Monkey OCR Pro 1.2B | 1.2B | 83.89 | 94.31 | 93.07 | 90.42 |
| Mineru 2.5 | 1.2B | 87.90 | 95.94 | 93.25 | 92.36 |
| Mineru 2.5 Pro | 1.2B | 92.46 | 97.24 | 93.98 | 94.56 |
| Paddle VL | 0.9B | 90.57 | 96.87 | 94.34 | 93.93 |
| Paddle VL 1.5 | 0.9B | 90.79 | 97.28 | 94.56 | 94.21 |
| GLM-OCR | 0.9B | 93.71 | 97.74 | 96.44 | 95.96 |
| MOSS-OCR-0.3B | 0.3B | 90.33 | 95.56 | 95.01 | 93.63 |

### In-house Patent-domain Benchmark

| Model | Params | Table (TEDS) | Math (CDM) | Overall |
|-------|--------|--------------|------------|---------|
| dolphin | 0.3B | 75.97 | 94.36 | 85.17 |
| Monkey OCR Pro 1.2B | 1.2B | 78.39 | 93.01 | 85.70 |
| Mineru 2.5 | 1.2B | 84.27 | 95.28 | 89.78 |
| Mineru 2.5 Pro | 1.2B | 87.97 | 96.56 | 92.27 |
| Paddle VL | 0.9B | 85.27 | 94.85 | 90.06 |
| Paddle VL 1.5 | 0.9B | 81.76 | 94.72 | 88.24 |
| GLM-OCR | 0.9B | 86.58 | 96.07 | 91.33 |
| MOSS-OCR-0.3B | 0.3B | 91.64 | 95.34 | 93.49 |

### Inference Speed on a Single RTX 4090

vLLM serving throughput.

| Model | Params | QPS (it/s) |
|-------|--------|------------|
| Mineru 2.5 | 1.2B | 29.49 |
| MOSS-OCR-0.3B | 0.3B | 58.77 |

---

## Requirements

- Python >= 3.12. [uv](https://github.com/astral-sh/uv) is recommended.
- CUDA-capable GPU for accelerated local inference and vLLM serving.
- vLLM serving requires the bundled adapter script so vLLM can register the MOSS model.

See [pyproject.toml](pyproject.toml) for pinned runtime dependencies.

---

## Model Weights

| Model | Download | Precision |
|-------|----------|-----------|
| Hiro-MOSS-OCR-0.3B | [PatSnap/Hiro-MOSS-OCR-0.3B](https://huggingface.co/PatSnap/Hiro-MOSS-OCR-0.3B) | FP32 / BF16 |

Use the Hugging Face repo id `PatSnap/Hiro-MOSS-OCR-0.3B` directly, or download the checkpoint to a local directory for offline deployments. The CUDA Graph and vLLM examples below accept either form as `MODEL_PATH`.

---

## Installation

Install from source:

```bash
git clone https://github.com/patsnap/Hiro-MOSS-OCR
cd Hiro-MOSS-OCR

uv python pin 3.12
uv venv .venv
source .venv/bin/activate
uv sync

# Copy MOSS patches into the installed vLLM package.
bash scripts/vllm_adapter.sh
```

`scripts/vllm_adapter.sh` copies the matching files from `moss_ocr/static/vllm_patches/` into the installed `vllm` package. Run it after `uv sync`, and rerun it if you reinstall or upgrade vLLM.

---

## Usage

### 1. Quick Call with Transformers `AutoModelForCausalLM`

For a quick smoke test, load and call the model directly with Hugging Face Transformers:

> This path is simple but relatively slow. Use it for quick trials, functional checks, or small single-image calls. For production serving, higher throughput, or batch inference, prefer the CUDA Graph or vLLM paths below. Keep the quick path on one GPU; it is not optimized for automatic multi-GPU module splitting.

```python
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # Set before importing torch.

import torch
from transformers import AutoModelForCausalLM

model_id = "PatSnap/Hiro-MOSS-OCR-0.3B"
img_path = "/path/to/your/image.png"
task = "text"  # "math" | "table" | "text"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map={"": 0},
).eval()

with torch.inference_mode():
    texts = model.generate(img_path, task=task)
print(texts[0])
```

The same quick path is available through the bundled example:

```bash
uv run python moss_ocr/examples/run_with_transformers.py \
  --model_path PatSnap/Hiro-MOSS-OCR-0.3B \
  --task text \
  --img_path /path/to/your/image.png
```

### 2. Local Inference with CUDA Graph + Transformers

Use `MOSSv1d6Runner` for single-process local inference:

```python
from moss_ocr.inferer.cuda_graph import MOSSv1d6Runner

model_path = "PatSnap/Hiro-MOSS-OCR-0.3B"
# Or: model_path = "/path/to/Hiro-MOSS-OCR-0.3B"
runner = MOSSv1d6Runner(model_path=model_path)

img_path = "/path/to/your/image.png"
task = "text"  # "math" | "table" | "text"

output = runner.run(img=img_path, task=task)
print(output)
```

The same path is available through the bundled example:

```bash
uv run python moss_ocr/examples/run_with_cuda_graph.py \
  --model_path PatSnap/Hiro-MOSS-OCR-0.3B \
  --task text \
  --img_path /path/to/your/image.png
```

### 3. vLLM Server with an OpenAI-compatible Client

First, start vLLM with either the Hugging Face repo id or a local model
checkpoint:

```bash
# Make sure `bash scripts/vllm_adapter.sh` has been run in this environment.
export MODEL_PATH=PatSnap/Hiro-MOSS-OCR-0.3B
# Or: export MODEL_PATH=/path/to/Hiro-MOSS-OCR-0.3B

uv run vllm serve "$MODEL_PATH" \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 16384 \
  --port 8088 \
  --served-model-name moss-v1d6-0.3b
```

Then call the server with `MOSSOCRv1d6vLLMRunner`. The `url` must include the `/v1` suffix:

```python
from moss_ocr.inferer.vllm import MOSSOCRv1d6vLLMRunner

runner = MOSSOCRv1d6vLLMRunner(url="http://0.0.0.0:8088/v1")

img_path = "/path/to/your/image.png"
task = "text"  # "math" | "table" | "text"

response = runner.run(img=img_path, task=task)
print(response.result if response.is_succeed else response.error_message)
```

CLI example:

```bash
uv run python moss_ocr/examples/run_with_vllm.py \
  --url http://0.0.0.0:8088/v1 \
  --task text \
  --img_path /path/to/your/image.png
```

The default `--served-model-name` should match the client's model name, `moss-v1d6-0.3b`. If you change the served name, pass `model_path="<your-served-name>"` when constructing `MOSSOCRv1d6vLLMRunner`.

### 4. Web Demo

The Gradio web demo runs the local CUDA Graph backend in the demo process. It accepts either the Hugging Face repo id or a local checkpoint path as `--model_path`:

```bash
uv run python moss_ocr/deploy/moss_ocr_demo.py \
  --model_path PatSnap/Hiro-MOSS-OCR-0.3B \
  --host 0.0.0.0 \
  --port 7788
```

Then open [http://127.0.0.1:7788](http://127.0.0.1:7788), or visit the server IP directly when binding to `0.0.0.0`. You can upload an image, choose formula, table, or text OCR from the task selector, or expand **Examples** to load the bundled sample images from `moss_ocr/static/img_examples/`.

For longer generations, increase `--max_length`; for larger local batches, adjust `--max_batch_size`.

---

## Notes

- OCR output can be inaccurate or incomplete. Review results before using them in legal, compliance, archival, accessibility, customer-facing, or other high-stakes workflows.
- Make sure you have the required rights and permissions for any images or documents you process with this project.
- For full terms and limitations, read the [disclaimer](docs/DISCLAIMER.md).

---

## Copyright Notice

Copyright (c) 2026 Patsnap. All rights reserved except as expressly licensed under the applicable license terms.

Hiro-MOSS-OCR, Patsnap, and any associated names, logos, product names, service names, designs, and slogans are trademarks or registered trademarks of Patsnap or its affiliates. No trademark license is granted under the open source license or any model license unless expressly stated.

---

## Acknowledgements

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [MinerU](https://github.com/opendatalab/MinerU)
- [GLM-OCR](https://github.com/zai-org/GLM-OCR)
- [Dolphin](https://github.com/bytedance/Dolphin)
- [Monkey-OCR](https://github.com/yuliang-liu/MonkeyOCR)
- [Smol-Docling](https://huggingface.co/docling-project/SmolDocling-256M-preview)
