# MOSS🍀：面向结构化标记序列的多模态 OCR

[English](README.md) | [简体中文](README_zh.md)

MOSS 是一个基于 **50M+ 训练数据从零训练**、面向块级文档理解的多模态 OCR 模型。它可以将文档图像区域转换为结构化标记：公式输出为 LaTeX，表格输出为 HTML，正文输出为 Markdown。模型支持日文、中文和英文。

项目地址：[github.com/patsnap/Hiro-MOSS-OCR](https://github.com/patsnap/Hiro-MOSS-OCR)

---

## 新闻与更新

<details>
<summary>近期更新</summary>

- **2026-05-26** - Hiro-MOSS-OCR-0.3B 已在 [Hugging Face](https://huggingface.co/PatSnap/Hiro-MOSS-OCR-0.3B) 开放。
- **2026-05-26** - 仓库已提供本地 CUDA Graph 推理和 vLLM 服务部署示例。

</details>

---

## 亮点

- **从零训练，训练数据 50M+：** 面向结构化 OCR 和文档图像理解场景专门构建。
- **结构化输出：** 支持公式识别、表格还原和正文抽取，并为不同任务生成对应的标记格式。
- **模型轻量：** 总参数量约 **320.8M**。
- **支持任意分辨率图像：** 基于 NaViT 风格视觉编码，并使用 2D RoPE。
- **两种推理方式：** 支持本地 Transformers/CUDA Graph 推理，也支持通过 vLLM 部署为 OpenAI 兼容服务。

---

## 模型概览

| 模块 | 说明 |
|------|------|
| 训练 | 基于 **50M+** 训练数据从零训练，支持任意分辨率图像 |
| 编码器（约 90M） | NaViT with 2D RoPE |
| 连接器（约 13.5M） | SwiGLU with patch merger |
| 解码器（约 216.6M） | Transformer decoder with pre-norm, RoPE, GQA, and SwiGLU |
| **总参数量** | **约 320.8M** |

## 支持任务

| 任务 | 输出格式 |
|------|----------|
| `math` | LaTeX |
| `table` | HTML |
| `text` | Markdown |

**支持语言：** 日文、中文、英文。

---

## 相关文档

- [免责声明](docs/DISCLAIMER.md) - 使用条款、责任限制和数据处理责任说明。
- [许可证](LICENSE) - 源代码许可证。

---

## 评测结果

### OmniDocBench v1.5

使用真实版面标注进行评测。

| 模型 | 参数量 | Table (TEDS) | Math (CDM) | Text (Edit Similarity) | Overall |
|------|--------|--------------|------------|-------------------------|---------|
| dolphin | 0.3B | 77.08 | 93.88 | 90.96 | 87.31 |
| Monkey OCR Pro 1.2B | 1.2B | 83.89 | 94.31 | 93.07 | 90.42 |
| Mineru 2.5 | 1.2B | 87.90 | 95.94 | 93.25 | 92.36 |
| Mineru 2.5 Pro | 1.2B | 92.46 | 97.24 | 93.98 | 94.56 |
| Paddle VL | 0.9B | 90.57 | 96.87 | 94.34 | 93.93 |
| Paddle VL 1.5 | 0.9B | 90.79 | 97.28 | 94.56 | 94.21 |
| GLM-OCR | 0.9B | 93.71 | 97.74 | 96.44 | 95.96 |
| MOSS-OCR-0.3B | 0.3B | 90.33 | 95.56 | 95.01 | 93.63 |

### 内部专利领域评测

| 模型 | 参数量 | Table (TEDS) | Math (CDM) | Overall |
|------|--------|--------------|------------|---------|
| dolphin | 0.3B | 75.97 | 94.36 | 85.17 |
| Monkey OCR Pro 1.2B | 1.2B | 78.39 | 93.01 | 85.70 |
| Mineru 2.5 | 1.2B | 84.27 | 95.28 | 89.78 |
| Mineru 2.5 Pro | 1.2B | 87.97 | 96.56 | 92.27 |
| Paddle VL | 0.9B | 85.27 | 94.85 | 90.06 |
| Paddle VL 1.5 | 0.9B | 81.76 | 94.72 | 88.24 |
| GLM-OCR | 0.9B | 86.58 | 96.07 | 91.33 |
| MOSS-OCR-0.3B | 0.3B | 91.64 | 95.34 | 93.49 |

### 单张 RTX 4090 上的推理速度

基于 vLLM 服务的吞吐量。

| 模型 | 参数量 | QPS (it/s) |
|------|--------|------------|
| Mineru 2.5 | 1.2B | 29.49 |
| MOSS-OCR-0.3B | 0.3B | 58.77 |

---

## 环境要求

- Python >= 3.12，推荐使用 [uv](https://github.com/astral-sh/uv)。
- 如需加速本地推理或使用 vLLM 部署，需要支持 CUDA 的 GPU。
- 使用 vLLM 服务时，需要运行仓库内置的适配脚本，让 vLLM 能够注册 MOSS 模型。

固定依赖版本见 [pyproject.toml](pyproject.toml)。

---

## 模型权重

| 模型 | 下载地址 | 精度 |
|------|----------|------|
| Hiro-MOSS-OCR-0.3B | [PatSnap/Hiro-MOSS-OCR-0.3B](https://huggingface.co/PatSnap/Hiro-MOSS-OCR-0.3B) | FP32 / BF16 |

请先将 checkpoint 下载到本地目录，然后在下面的命令中将该目录作为 `MODEL_PATH` 使用。

---

## 安装

从源码安装：

```bash
git clone https://github.com/patsnap/Hiro-MOSS-OCR
cd Hiro-MOSS-OCR

uv python pin 3.12
uv venv .venv
source .venv/bin/activate
uv sync

# 将 MOSS 适配文件复制到当前环境中的 vLLM 包内。
bash scripts/vllm_adapter.sh
```

`scripts/vllm_adapter.sh` 会把 `moss_ocr/static/vllm_patches/` 中与当前固定 vLLM 版本匹配的文件复制到已安装的 `vllm` 包内。请在 `uv sync` 后运行该脚本；如果重新安装或升级 vLLM，也需要重新运行。

---

## 使用方式

### 1. 使用 CUDA Graph + Transformers 本地推理

可以使用 `MOSSv1d6Runner` 进行单进程本地推理：

```python
from moss_ocr.inferer.cuda_graph import MOSSv1d6Runner

model_path = "/path/to/Hiro-MOSS-OCR-0.3B"
runner = MOSSv1d6Runner(model_path=model_path)

img_path = "/path/to/your/image.png"
task = "text"  # "math" | "table" | "text"

output = runner.run(img=img_path, task=task)
print(output)
```

也可以直接运行仓库内置示例：

```bash
uv run python moss_ocr/examples/run_with_cuda_graph.py \
  --model_path /path/to/Hiro-MOSS-OCR-0.3B \
  --task text \
  --img_path /path/to/your/image.png
```

### 2. 使用 vLLM 服务和 OpenAI 兼容客户端

首先使用 Hugging Face repo id 或本地 checkpoint 启动 vLLM：

```bash
export MODEL_PATH=PatSnap/Hiro-MOSS-OCR-0.3B
# 或：export MODEL_PATH=/path/to/Hiro-MOSS-OCR-0.3B

uv run vllm serve "$MODEL_PATH" \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 16384 \
  --port 8088 \
  --served-model-name moss-v1d6-0.3b
```

然后用 `MOSSOCRv1d6vLLMRunner` 调用服务。注意 `url` 需要包含 `/v1` 后缀：

```python
from moss_ocr.inferer.vllm import MOSSOCRv1d6vLLMRunner

runner = MOSSOCRv1d6vLLMRunner(url="http://127.0.0.1:8088/v1")

img_path = "/path/to/your/image.png"
task = "text"  # "math" | "table" | "text"

response = runner.run(img=img_path, task=task)
print(response.result if response.is_succeed else response.error_message)
```

CLI 示例：

```bash
uv run python moss_ocr/examples/run_with_vllm.py \
  --url http://127.0.0.1:8088/v1 \
  --task text \
  --img_path /path/to/your/image.png
```

默认的 `--served-model-name` 应与客户端模型名 `moss-v1d6-0.3b` 保持一致。如果你修改了服务端名称，请在构造 `MOSSOCRv1d6vLLMRunner` 时传入 `model_path="<your-served-name>"`。

---

## 注意事项

- OCR 输出可能存在错误、不完整或结构不准确的情况。用于法律、合规、归档、无障碍、客户交付或其他高风险场景前，请务必人工复核。
- 请确认你对待处理的图片或文档拥有必要的权利、授权和处理依据。
- 完整条款和限制请阅读 [免责声明](docs/DISCLAIMER.md)。

---

## 版权声明

Copyright (c) 2026 Patsnap. 除非适用许可条款明确授权，保留所有权利。

Hiro-MOSS-OCR、Patsnap 以及任何相关名称、徽标、产品名称、服务名称、设计和标语均为 Patsnap 或其关联公司的商标或注册商标。除非明确说明，开源许可证或任何模型许可证均不授予任何商标许可。

---

## 致谢

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [MinerU](https://github.com/opendatalab/MinerU)
- [GLM-OCR](https://github.com/zai-org/GLM-OCR)
- [Dolphin](https://github.com/bytedance/Dolphin)
- [Monkey-OCR](https://github.com/yuliang-liu/MonkeyOCR)
- [Smol-Docling](https://huggingface.co/docling-project/SmolDocling-256M-preview)
