# Whisper 加速方案调研报告

> 调研日期: 2026-07-22
> 目标: 为 Automatic-Subtitle-Generator 项目选择 Whisper 后端实现方案
> 原则: 不重复造轮子，只做胶水代码，基于 SttRunner Protocol 薄包装

---

## 1. 调研范围

| # | 方案 | 类型 | 调研状态 |
|---|------|------|----------|
| 1 | faster-whisper | CTranslate2 后端 | ✅ 详细 |
| 2 | whisper.cpp | C/C++ 原生 | ✅ 详细 |
| 3 | insanely-fast-whisper | ONNX/Transformers | ✅ 详细 |
| 4 | whisperX | 词级时间戳 + 说话人 | ✅ 详细 |
| 5 | sherpa-onnx | C++/ONNX 跨平台 | ✅ 详细 |
| 6 | mlx-whisper | Apple MLX 框架 | ✅ 详细 |
| 7 | faster-whisper-xxl | 批处理包装 | ✅ 评估 |
| 8 | whisper-medusa | 多头解码 | ✅ 评估 |
| 9 | whisper-distillation | 蒸馏方案 | ✅ 评估 |
| 10 | Parakeet (NVIDIA) | NVIDIA ASR | ✅ 评估 |

---

## 2. 主流方案详细调研

### 2.1 faster-whisper (SYSTRAN/faster-whisper)

| 指标 | 数据 |
|------|------|
| **GitHub Stars** | 24,438 ⭐ |
| **最新 Release** | v1.2.2 (2026年内持续更新) |
| **最后提交** | 2025-11-19 |
| **License** | MIT |
| **语言** | Python |
| **PyPI** | `faster-whisper` (v1.2.1) |
| **Python 要求** | ≥ 3.9 |

**平台支持:**
- ✅ macOS (Intel & ARM) — CPU
- ✅ Linux — CPU / CUDA
- ✅ Windows — CPU / CUDA
- ⚠️ Apple Silicon GPU: 无原生 Metal 支持（走 CPU + NEON）

**性能 (相对原始 whisper):**
- GPU (RTX 3070 Ti, large-v2, fp16): **~2x** 速度，比 openai/whisper 快 2 倍以上
- CPU (i7-12700K, 8线程, int8): **~4x** 速度
- 内存使用比 openai/whisper 减少 30-50%
- 支持 batch 推理，batch_size=8 时 GPU 上可达 **8-10x** 加速

**输出格式:**
- ✅ Segment 级时间戳 (start, end, text)
- ✅ Word 级时间戳 (`word_timestamps=True`)
- ✅ 语言检测概率
- ✅ 流式输出 (generator)

**安装:**
```bash
pip install faster-whisper
```

**最小可用代码:**
```python
from faster_whisper import WhisperModel

model = WhisperModel("large-v3", device="cpu", compute_type="int8")
segments, info = model.transcribe("audio.wav", beam_size=5)

for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
```

**模型大小:**
| 模型 | 磁盘 | 内存 |
|------|------|------|
| tiny | 75 MiB | ~273 MB |
| base | 142 MiB | ~388 MB |
| small | 466 MiB | ~852 MB |
| medium | 1.5 GiB | ~2.1 GB |
| large-v3 | 2.9 GiB | ~3.9 GB |
| distil-large-v3 | ~1.5 GiB | ~2 GB |

**与项目兼容性:**
- ✅ 纯 Python，API 简洁
- ✅ 返回 segment 时间戳，直接映射到 `Segment` NamedTuple
- ✅ 支持 `language` 参数
- ✅ 流式输出可适配 `on_progress` 回调
- ✅ 自动设备检测 (cpu/cuda)
- ⚠️ 无原生 MPS/Metal 支持（Apple Silicon 上走 CPU）
- ⚠️ 模型需从 HuggingFace 下载（CTranslate2 格式）

---

### 2.2 whisper.cpp (ggerganov/whisper.cpp → ggml-org/whisper.cpp)

| 指标 | 数据 |
|------|------|
| **GitHub Stars** | 52,120 ⭐ |
| **最新 Release** | v1.9.1 (活跃维护) |
| **最后提交** | 2026-07-11 |
| **License** | MIT |
| **语言** | C/C++ (核心) + 多语言绑定 |
| **PyPI (Python 绑定)** | `pywhispercpp` v1.5.0 (2026-05-30) |

**平台支持:**
- ✅ macOS (Intel & ARM) — **Metal GPU 原生加速**
- ✅ Linux — CPU / CUDA / Vulkan / OpenVINO
- ✅ Windows — CPU / CUDA / Vulkan
- ✅ iOS / Android / WebAssembly / Raspberry Pi
- ✅ Apple Neural Engine (Core ML)
- ✅ AMD ROCm, Ascend NPU, Moore Threads

**性能 (相对原始 whisper):**
- Apple Silicon (M1, small, Metal): **~3-5x** 实时
- Apple Silicon (M2 Pro, medium, Metal): **~7-10x** 实时
- Apple Silicon (M4 Max, large-v3, Metal): **~8-15x** 实时
- Apple Silicon (M4 Max, medium, Metal): **~15-25x** 实时
- 支持 Q4_0/Q8_0 量化，进一步加速

**输出格式:**
- ✅ Segment 级时间戳
- ✅ Token 级时间戳
- ✅ VAD (Voice Activity Detection)
- ⚠️ Python 绑定 (pywhispercpp) API 较底层

**安装 (Python):**
```bash
pip install pywhispercpp
```

**最小可用代码 (Python):**
```python
from pywhispercpp.model import Model

model = Model('base.en')
segments = model.transcribe('file.wav')
for segment in segments:
    print(segment.text)
```

**模型大小 (ggml 格式):**
| 模型 | 磁盘 | 内存 |
|------|------|------|
| tiny | 75 MiB | ~273 MB |
| base | 142 MiB | ~388 MB |
| small | 466 MiB | ~852 MB |
| medium | 1.5 GiB | ~2.1 GB |
| large-v3 | 2.9 GiB | ~3.9 GB |
| large-v3-turbo | ~1.9 GiB | ~2.5 GB |

**与项目兼容性:**
- ✅ Apple Silicon 上性能最佳（Metal 原生）
- ✅ 跨平台最广
- ✅ 量化模型小，适合分发
- ⚠️ Python 绑定 API 不如 faster-whisper 简洁
- ⚠️ 模型需转换为 ggml 格式（或下载预转换版本）
- ⚠️ 进度回调需要额外封装

---

### 2.3 insanely-fast-whisper (Vaibhavs10/insanely-fast-whisper)

| 指标 | 数据 |
|------|------|
| **GitHub Stars** | 12,993 ⭐ |
| **最后提交** | 2024-06-06 (⚠️ 维护不活跃) |
| **License** | Apache-2.0 |
| **PyPI** | `insanely-fast-whisper` v0.0.15 |

**平台支持:**
- ✅ macOS (MPS)
- ✅ Linux (CUDA)
- ✅ Windows (CUDA)
- ⚠️ 仅 CLI 工具，非库

**性能:**
- A100 80GB, large-v3, fp16 + Flash Attention 2 + batch=24: **150分钟音频 < 98秒** (~90x 实时)
- T4 GPU: 约 30-50x 实时
- ⚠️ 高度依赖高端 GPU

**输出格式:**
- ✅ Segment 级时间戳
- ✅ Word 级时间戳
- ✅ 说话人识别 (pyannote)
- ✅ JSON 输出

**安装:**
```bash
pipx install insanely-fast-whisper
```

**最小可用代码:**
```bash
insanely-fast-whisper --file-name audio.wav --model-name openai/whisper-large-v3
```

**与项目兼容性:**
- ⚠️ 主要是 CLI 工具，不适合做库集成
- ⚠️ 维护不活跃（最后提交 2024-06）
- ⚠️ 依赖 Flash Attention 2，安装复杂
- ⚠️ 需要 HuggingFace token 用于说话人识别
- ❌ 不推荐作为 SttRunner 后端

---

### 2.4 whisperX (m-bain/whisperX)

| 指标 | 数据 |
|------|------|
| **GitHub Stars** | 23,178 ⭐ |
| **最新 Release** | v3.8.6 (2026-05-25) |
| **最后提交** | 2026-07-13 (活跃) |
| **License** | BSD-2-Clause |
| **PyPI** | `whisperx` v3.8.6 |
| **Python 要求** | ≥ 3.10, < 3.14 |

**平台支持:**
- ✅ macOS (CPU / MPS)
- ✅ Linux (CPU / CUDA)
- ✅ Windows (CPU / CUDA)

**性能:**
- large-v2, batch=16, GPU: **~70x** 实时
- ⚠️ 词级对齐增加额外开销
- ⚠️ 说话人识别需要额外模型

**输出格式:**
- ✅ Segment 级时间戳
- ✅ **Word 级时间戳** (通过 wav2vec2 强制对齐)
- ✅ 说话人标签
- ✅ VAD 预处理

**安装:**
```bash
pip install whisperx
```

**最小可用代码:**
```python
import whisperx

model = whisperx.load_model("large-v2", "cuda", compute_type="float16")
audio = whisperx.load_audio("audio.wav")
result = model.transcribe(audio, batch_size=16)

# 词级对齐
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device="cuda")
result = whisperx.align(result["segments"], model_a, metadata, audio, "cuda")
```

**与项目兼容性:**
- ✅ 活跃维护
- ✅ 功能丰富（词级时间戳 + 说话人）
- ⚠️ 依赖链复杂（faster-whisper + pyannote + wav2vec2）
- ⚠️ 安装体积大
- ⚠️ 词级对齐对中文支持有限
- ⚠️ 架构重，不适合做薄包装

---

### 2.5 sherpa-onnx (k2-fsa/sherpa-onnx)

| 指标 | 数据 |
|------|------|
| **GitHub Stars** | 13,708 ⭐ |
| **最新 Release** | 持续发布 (2026-07 活跃) |
| **最后提交** | 2026-07-22 (极活跃) |
| **License** | Apache-2.0 |
| **PyPI** | `sherpa-onnx` v1.13.4 |
| **Python 要求** | ≥ 3.7 |

**平台支持:**
- ✅ macOS (ARM + Intel)
- ✅ Linux (x86_64, ARM, RISC-V)
- ✅ Windows
- ✅ iOS / Android / HarmonyOS
- ✅ Raspberry Pi, RK NPU, Ascend NPU, Axera NPU
- ✅ 12 种编程语言绑定

**性能:**
- 跨平台一致性能
- int8 量化模型，内存占用低
- 支持流式和非流式推理

**输出格式:**
- ✅ Segment 级时间戳
- ✅ 说话人识别
- ✅ VAD
- ⚠️ API 较底层

**安装:**
```bash
pip install sherpa-onnx
```

**最小可用代码:**
```python
import sherpa_onnx
import soundfile as sf

recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
    encoder="tiny.en-encoder.int8.onnx",
    decoder="tiny.en-decoder.int8.onnx",
    tokens="tiny.en-tokens.txt",
)
audio, sample_rate = sf.read("audio.wav", dtype="float32", always_2d=True)
stream = recognizer.create_stream()
stream.accept_waveform(sample_rate, audio)
recognizer.decode_stream(stream)
result = stream.get_result()
print(result.text)
```

**与项目兼容性:**
- ✅ 极活跃维护
- ✅ 跨平台最广（含嵌入式）
- ✅ 离线运行，无需联网
- ⚠️ API 较底层，需要手动管理 stream
- ⚠️ 模型需从 sherpa-onnx 发布页下载（ONNX 格式）
- ⚠️ 进度回调需要额外封装

---

### 2.6 mlx-whisper (ml-explore/mlx-examples)

| 指标 | 数据 |
|------|------|
| **GitHub Stars** | 8,844 ⭐ (mlx-examples 总计) |
| **最后提交** | 2026-04-06 |
| **License** | MIT |
| **PyPI** | `mlx-whisper` v0.4.3 |
| **Python 要求** | ≥ 3.8 |

**平台支持:**
- ✅ **macOS Apple Silicon only** (MLX 框架)
- ❌ Linux (无官方支持)
- ❌ Windows (无官方支持)

**性能 (Apple Silicon):**
- M1/M2/M3/M4 上通过 MLX 统一内存架构高效运行
- large-v3 在 M1 Ultra 上约 **10-20x** 实时
- 支持 fp16 和 4-bit/8-bit 量化
- 统一内存，可利用全部系统内存

**输出格式:**
- ✅ Segment 级时间戳
- ✅ Word 级时间戳 (`word_timestamps=True`)
- ✅ 语言检测
- ✅ 多种输出格式 (txt, srt, vtt, tsv, json)

**安装:**
```bash
pip install mlx-whisper
```

**最小可用代码:**
```python
import mlx_whisper

result = mlx_whisper.transcribe(
    "audio.wav",
    path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
    language="zh",
)
print(result["text"])

# 带 segment 时间戳
for segment in result["segments"]:
    print(f"[{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text']}")
```

**模型大小 (MLX 格式):**
| 模型 | 磁盘 (fp16) | 磁盘 (4-bit) |
|------|-------------|--------------|
| tiny | ~75 MiB | ~25 MiB |
| base | ~142 MiB | ~45 MiB |
| small | ~466 MiB | ~150 MiB |
| medium | ~1.5 GiB | ~480 MiB |
| large-v3 | ~2.9 GiB | ~930 MiB |

**与项目兼容性:**
- ✅ API 极其简洁
- ✅ 返回格式直接可用
- ✅ Apple Silicon 性能优秀
- ✅ 与项目现有 Qwen3-ASR (MLX) 架构一致
- ⚠️ **仅支持 macOS Apple Silicon**
- ⚠️ 依赖 mlx 框架（仅 macOS）
- ⚠️ 模型需 MLX 格式（HuggingFace 有预转换）

---

### 2.7 faster-whisper-xxl (社区批处理包装)

| 指标 | 数据 |
|------|------|
| **GitHub Stars** | 无独立高星仓库 |
| **类型** | faster-whisper 的批处理 CLI 包装 |
| **维护** | 社区分散，无统一标准 |

**评估:**
- 本质是 faster-thisper 的批处理脚本包装
- 功能: 监控文件夹、自动转录、输出 SRT/VTT
- 项目已有 orchestrator 批处理逻辑，**无需此方案**
- ❌ 不推荐（与项目功能重复）

---

### 2.8 whisper-medusa (aiola-lab/whisper-medusa)

| 指标 | 数据 |
|------|------|
| **GitHub Stars** | 860 ⭐ |
| **最后提交** | 2026-07-02 |
| **License** | MIT |

**性能:**
- 比原始 whisper 快 **~1.5x**
- 仅支持英语
- WER 略有下降 (4.1% vs 4%)

**评估:**
- ⚠️ 加速比有限 (仅 1.5x)
- ⚠️ 仅支持英语
- ⚠️ 需要训练自定义 Medusa heads
- ⚠️ 模型生态小
- ❌ 不适合多语言字幕场景

---

### 2.9 whisper-distillation (各类蒸馏方案)

| 指标 | 数据 |
|------|------|
| **GitHub Stars** | 分散，无统一方案 |
| **类型** | 学术研究为主 |

**评估:**
- distil-whisper (HuggingFace): 可用，但 faster-whisper 已支持
- 其他蒸馏方案多为论文代码，未产品化
- ⚠️ 生态分散，维护不一
- ❌ 不推荐作为独立后端

---

### 2.10 Parakeet / NVIDIA NeMo

| 指标 | 数据 |
|------|------|
| **GitHub Stars** | NeMo 高星 (具体未获取) |
| **License** | Apache-2.0 |
| **类型** | NVIDIA 官方 ASR 框架 |

**评估:**
- Parakeet TDT 0.6B: 极快，但非 Whisper 架构
- 社区有 ONNX 导出版本 (achetronic/parakeet: 217 stars)
- ⚠️ 非 Whisper 模型，不兼容 Whisper 生态
- ⚠️ 依赖 NVIDIA 生态
- ⚠️ 中文支持有限
- ❌ 不适合（项目定位 Whisper 后端）

---

## 3. 综合对比矩阵

| 方案 | Stars | 维护 | macOS ARM | Linux | Windows | Apple 性能 | 安装难度 | 时间戳 | 推荐度 |
|------|-------|------|-----------|-------|---------|-----------|----------|--------|--------|
| **faster-whisper** | 24.4k | ✅ | ✅ CPU | ✅ | ✅ | 中 (CPU) | ⭐ 极易 | ✅ | ⭐⭐⭐⭐⭐ |
| **whisper.cpp** | 52.1k | ✅ | ✅ Metal | ✅ | ✅ | **高** | ⭐⭐ 易 | ✅ | ⭐⭐⭐⭐⭐ |
| **insanely-fast-whisper** | 13k | ❌ | ✅ MPS | ✅ | ✅ | 高 | ⭐⭐⭐ 中 | ✅ | ⭐⭐ |
| **whisperX** | 23.2k | ✅ | ✅ | ✅ | ✅ | 中 | ⭐⭐⭐⭐ 难 | ✅✅ | ⭐⭐⭐ |
| **sherpa-onnx** | 13.7k | ✅ | ✅ | ✅ | ✅ | 中 | ⭐⭐⭐ 中 | ✅ | ⭐⭐⭐⭐ |
| **mlx-whisper** | 8.8k | ✅ | ✅ **MLX** | ❌ | ❌ | **极高** | ⭐ 极易 | ✅ | ⭐⭐⭐⭐⭐ |
| faster-whisper-xxl | - | ❌ | - | - | - | - | - | - | ❌ |
| whisper-medusa | 860 | ✅ | ✅ | ✅ | ✅ | 低 | ⭐⭐⭐⭐ 难 | ✅ | ⭐⭐ |
| whisper-distill | 分散 | ❌ | - | - | - | - | - | - | ❌ |
| Parakeet/NeMo | - | ✅ | ❌ | ✅ | ✅ | - | ⭐⭐⭐⭐ 难 | ✅ | ⭐⭐ |

---

## 4. Top 3 推荐方案

### 🥇 第一名: faster-whisper — 通用首选

**推荐理由:**
1. **最广泛的平台支持**: macOS/Linux/Windows 全平台，CPU/CUDA 自动检测
2. **API 最简洁**: 返回 segment 时间戳，直接映射到项目 Protocol
3. **安装最简单**: `pip install faster-whisper`，无需编译
4. **维护活跃**: SYSTRAN 官方支持，持续更新
5. **性能优秀**: 比原始 whisper 快 4x，内存占用低
6. **模型生态丰富**: HuggingFace 上大量 CTranslate2 格式模型

**劣势:**
- Apple Silicon 上无 Metal 加速（走 CPU）
- 模型下载需要 HuggingFace 访问

**适用场景:** 跨平台部署、Linux 服务器、Windows 用户

---

### 🥈 第二名: whisper.cpp (pywhispercpp) — Apple Silicon 性能之王

**推荐理由:**
1. **Apple Silicon 性能最佳**: Metal GPU 原生加速，M4 Max 上 large-v3 可达 15x 实时
2. **Star 数最高**: 52k+ stars，社区最大
3. **跨平台最广**: 从嵌入式到服务器全覆盖
4. **量化成熟**: Q4_0/Q8_0 量化，模型体积小
5. **离线友好**: 模型一次下载，永久离线使用

**劣势:**
- Python 绑定 API 不如 faster-whisper 简洁
- 进度回调需要额外封装
- 模型需 ggml 格式

**适用场景:** macOS 桌面应用、追求极致性能、嵌入式部署

---

### 🥉 第三名: mlx-whisper — Apple Silicon 原生体验

**推荐理由:**
1. **与项目现有架构一致**: 已有 Qwen3-ASR (MLX) 后端
2. **API 极其简洁**: 一行代码完成转录
3. **Apple Silicon 性能极佳**: MLX 统一内存架构
4. **安装简单**: `pip install mlx-whisper`
5. **模型生态好**: HuggingFace MLX Community 大量预转换模型

**劣势:**
- ⚠️ **仅支持 macOS Apple Silicon**
- 无法跨平台

**适用场景:** macOS 专用版本、与 Qwen3-ASR 并列的 Apple 优化后端

---

## 5. 实现框架设计

### 5.1 推荐实现顺序

```
Phase 1: FasterWhisperRunner (faster-whisper) — 通用后端
    ↓
Phase 2: MlxWhisperRunner (mlx-whisper) — Apple 优化后端
    ↓
Phase 3: WhisperCppRunner (pywhispercpp) — 高性能/嵌入式后端
```

### 5.2 FasterWhisperRunner 实现框架

```python
"""Whisper runner — 基于 faster-whisper (CTranslate2 后端).

跨平台通用 Whisper 后端，支持 CPU 和 CUDA GPU。
模型自动从 HuggingFace 下载 CTranslate2 格式。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from subtool.core.protocol import (
    HardwareCapabilities,
    Segment,
    SttRunner,
    TranscriptionError,
    TranscriptionResult,
)
from subtool.hardware.detect import detect_hardware


# 可用模型预设
MODELS = {
    "tiny":       "tiny",
    "base":       "base",
    "small":      "small",
    "medium":     "medium",
    "large-v2":   "large-v2",
    "large-v3":   "large-v3",
    "large-v3-turbo": "large-v3-turbo",
    "distil-large-v3": "distil-whisper/distil-large-v3",
}

DEFAULT_MODEL = "large-v3-turbo"  # 默认使用 turbo 模型（快且准）

# 模型内存需求 (bytes)
MODEL_MEMORY = {
    "tiny":       512 * 1024 * 1024,
    "base":       768 * 1024 * 1024,
    "small":      1536 * 1024 * 1024,
    "medium":     3 * 1024 * 1024 * 1024,
    "large-v2":   6 * 1024 * 1024 * 1024,
    "large-v3":   6 * 1024 * 1024 * 1024,
    "large-v3-turbo": 3500 * 1024 * 1024,
    "distil-large-v3": 3 * 1024 * 1024 * 1024,
}


class FasterWhisperRunner:
    """基于 faster-whisper 的 Whisper 转录后端.

    特点:
    - 跨平台: macOS/Linux/Windows
    - 设备自动检测: CPU / CUDA
    - 支持 int8 量化（CPU）和 float16（GPU）
    - 流式输出，低内存占用
    """

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL,
        device: str | None = None,
        compute_type: str | None = None,
        device_index: int = 0,
        cpu_threads: int = 4,
        num_workers: int = 1,
    ) -> None:
        self._model_size = model_size
        self._device = device  # "auto" | "cpu" | "cuda"
        self._compute_type = compute_type  # "int8" | "float16" | "int8_float16"
        self._device_index = device_index
        self._cpu_threads = cpu_threads
        self._num_workers = num_workers
        self._model = None  # 延迟加载

    @property
    def name(self) -> str:
        return f"whisper-faster-{self._model_size}"

    def capabilities(self) -> HardwareCapabilities:
        return detect_hardware()

    def can_run(self, source: Path, hardware: HardwareCapabilities) -> bool:
        """检查内存是否足够加载模型."""
        mem_needed = MODEL_MEMORY.get(self._model_size, 6 * 1024 * 1024 * 1024)
        return hardware.memory_bytes >= mem_needed

    def _load_model(self):
        """延迟加载模型."""
        if self._model is not None:
            return

        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise TranscriptionError(
                "faster-whisper 未安装。请运行: pip install faster-whisper"
            )

        # 自动选择设备
        device = self._device
        if device is None or device == "auto":
            hw = detect_hardware()
            device = "cuda" if hw.device == "cuda" else "cpu"

        # 自动选择计算类型
        compute_type = self._compute_type
        if compute_type is None:
            compute_type = "float16" if device == "cuda" else "int8"

        try:
            self._model = WhisperModel(
                self._model_size,
                device=device,
                compute_type=compute_type,
                device_index=self._device_index,
                cpu_threads=self._cpu_threads,
                num_workers=self._num_workers,
            )
        except Exception as e:
            raise TranscriptionError(f"加载 Whisper 模型失败: {e}") from e

    def transcribe(
        self,
        source: Path,
        *,
        language: str | None = None,
        on_progress: Callable[[float], None] | None = None,
    ) -> TranscriptionResult:
        """转录单个音频文件."""
        self._load_model()

        try:
            segments_iter, info = self._model.transcribe(
                str(source),
                language=language,
                beam_size=5,
                vad_filter=True,  # VAD 过滤静音
                vad_parameters=dict(min_silence_duration_ms=500),
            )

            # 收集 segments
            segments: list[Segment] = []
            total_duration = info.duration

            for seg in segments_iter:
                segments.append(Segment(
                    start=float(seg.start),
                    end=float(seg.end),
                    text=seg.text.strip(),
                    confidence=seg.avg_logprob if hasattr(seg, 'avg_logprob') else None,
                ))
                # 进度回调
                if on_progress and total_duration > 0:
                    on_progress(min(seg.end / total_duration, 1.0))

            return TranscriptionResult(
                segments=segments,
                language=info.language,
                duration=total_duration,
            )

        except Exception as e:
            if isinstance(e, TranscriptionError):
                raise
            raise TranscriptionError(f"Whisper 转录失败: {e}") from e
```

### 5.3 MlxWhisperRunner 实现框架

```python
"""Whisper runner — 基于 mlx-whisper (Apple MLX 后端).

Apple Silicon 优化的 Whisper 后端，使用 MLX 框架。
仅支持 macOS Apple Silicon 平台。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

from subtool.core.protocol import (
    HardwareCapabilities,
    Segment,
    SttRunner,
    TranscriptionError,
    TranscriptionResult,
)
from subtool.hardware.detect import detect_hardware


# MLX 格式模型路径 (HuggingFace)
MODELS = {
    "tiny":           "mlx-community/whisper-tiny-mlx",
    "base":           "mlx-community/whisper-base-mlx",
    "small":          "mlx-community/whisper-small-mlx",
    "medium":         "mlx-community/whisper-medium-mlx",
    "large-v2":       "mlx-community/whisper-large-v2-mlx",
    "large-v3":       "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo-mlx",
    "large-v3-turbo-4bit": "mlx-community/whisper-large-v3-turbo-mlx-4bit",
}

DEFAULT_MODEL = "large-v3-turbo"


class MlxWhisperRunner:
    """基于 mlx-whisper 的 Apple Silicon 优化 Whisper 后端.

    特点:
    - 仅支持 macOS Apple Silicon
    - MLX 统一内存架构，可利用全部系统内存
    - API 简洁，与项目 Qwen3-ASR (MLX) 架构一致
    """

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL,
        path_or_hf_repo: str | None = None,
    ) -> None:
        self._model_size = model_size
        self._model_repo = path_or_hf_repo or MODELS.get(
            model_size, f"mlx-community/whisper-{model_size}-mlx"
        )

    @property
    def name(self) -> str:
        return f"whisper-mlx-{self._model_size}"

    def capabilities(self) -> HardwareCapabilities:
        return detect_hardware()

    def can_run(self, source: Path, hardware: HardwareCapabilities) -> bool:
        """检查是否在 Apple Silicon 上."""
        if sys.platform != "darwin":
            return False
        if hardware.device != "mps":
            return False
        # large-v3 需要至少 8GB 统一内存
        mem_needed = 8 * 1024 * 1024 * 1024 if "large" in self._model_size else 4 * 1024 * 1024 * 1024
        return hardware.memory_bytes >= mem_needed

    def transcribe(
        self,
        source: Path,
        *,
        language: str | None = None,
        on_progress: Callable[[float], None] | None = None,
    ) -> TranscriptionResult:
        """转录单个音频文件."""
        try:
            import mlx_whisper
        except ImportError:
            raise TranscriptionError(
                "mlx-whisper 未安装。请运行: pip install mlx-whisper"
            )

        try:
            result = mlx_whisper.transcribe(
                str(source),
                path_or_hf_repo=self._model_repo,
                language=language,
                verbose=False,
            )

            segments: list[Segment] = []
            for seg in result.get("segments", []):
                segments.append(Segment(
                    start=float(seg.get("start", 0.0)),
                    end=float(seg.get("end", 0.0)),
                    text=seg.get("text", "").strip(),
                    confidence=None,
                ))

            # mlx-whisper 不直接支持进度回调
            if on_progress:
                on_progress(1.0)

            duration = segments[-1].end if segments else 0.0

            return TranscriptionResult(
                segments=segments,
                language=result.get("language"),
                duration=duration,
            )

        except Exception as e:
            if isinstance(e, TranscriptionError):
                raise
            raise TranscriptionError(f"mlx-whisper 转录失败: {e}") from e
```

### 5.4 WhisperCppRunner 实现框架

```python
"""Whisper runner — 基于 whisper.cpp (pywhispercpp 绑定).

高性能 C/C++ 实现，Apple Silicon 上通过 Metal GPU 加速。
跨平台支持最广，从嵌入式到服务器均可使用。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from subtool.core.protocol import (
    HardwareCapabilities,
    Segment,
    SttRunner,
    TranscriptionError,
    TranscriptionResult,
)
from subtool.hardware.detect import detect_hardware


# whisper.cpp 模型名 (ggml 格式)
MODELS = {
    "tiny":       "tiny",
    "base":       "base",
    "small":      "small",
    "medium":     "medium",
    "large-v2":   "large-v2",
    "large-v3":   "large-v3",
    "large-v3-turbo": "large-v3-turbo",
}

DEFAULT_MODEL = "large-v3-turbo"


class WhisperCppRunner:
    """基于 whisper.cpp 的高性能 Whisper 后端.

    特点:
    - C/C++ 原生实现，性能极高
    - Apple Silicon Metal GPU 加速
    - 支持量化模型 (Q4_0, Q8_0)
    - 跨平台: macOS/Linux/Windows/嵌入式
    """

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL,
        model_path: Path | None = None,
        use_gpu: bool = True,
    ) -> None:
        self._model_size = model_size
        self._model_path = model_path
        self._use_gpu = use_gpu
        self._model = None

    @property
    def name(self) -> str:
        return f"whisper-cpp-{self._model_size}"

    def capabilities(self) -> HardwareCapabilities:
        return detect_hardware()

    def can_run(self, source: Path, hardware: HardwareCapabilities) -> bool:
        """whisper.cpp 对硬件要求最低."""
        mem_needed = 4 * 1024 * 1024 * 1024 if "large" in self._model_size else 2 * 1024 * 1024 * 1024
        return hardware.memory_bytes >= mem_needed

    def _load_model(self):
        """延迟加载模型."""
        if self._model is not None:
            return

        try:
            from pywhispercpp.model import Model
        except ImportError:
            raise TranscriptionError(
                "pywhispercpp 未安装。请运行: pip install pywhispercpp"
            )

        model_name = str(self._model_path) if self._model_path else self._model_size

        try:
            self._model = Model(
                model_name,
                n_threads=4 if self._use_gpu else 8,
                use_gpu=self._use_gpu,
            )
        except Exception as e:
            raise TranscriptionError(f"加载 whisper.cpp 模型失败: {e}") from e

    def transcribe(
        self,
        source: Path,
        *,
        language: str | None = None,
        on_progress: Callable[[float], None] | None = None,
    ) -> TranscriptionResult:
        """转录单个音频文件."""
        self._load_model()

        try:
            segments = self._model.transcribe(
                str(source),
                language=language or "auto",
            )

            result_segments: list[Segment] = []
            for seg in segments:
                result_segments.append(Segment(
                    start=float(getattr(seg, 'start', seg.t0) / 1000.0 if hasattr(seg, 't0') else 0.0),
                    end=float(getattr(seg, 'end', seg.t1) / 1000.0 if hasattr(seg, 't1') else 0.0),
                    text=seg.text.strip(),
                    confidence=None,
                ))

            duration = result_segments[-1].end if result_segments else 0.0

            if on_progress:
                on_progress(1.0)

            return TranscriptionResult(
                segments=result_segments,
                language=language,
                duration=duration,
            )

        except Exception as e:
            if isinstance(e, TranscriptionError):
                raise
            raise TranscriptionError(f"whisper.cpp 转录失败: {e}") from e
```

---

## 6. 胶水代码集成要点

### 6.1 与 SttRunner Protocol 的映射

| Protocol 方法 | faster-whisper | mlx-whisper | whisper.cpp |
|---------------|----------------|-------------|-------------|
| `name` | `whisper-faster-{size}` | `whisper-mlx-{size}` | `whisper-cpp-{size}` |
| `capabilities()` | `detect_hardware()` | `detect_hardware()` | `detect_hardware()` |
| `can_run()` | 内存检查 | 内存 + 平台检查 | 内存检查 |
| `transcribe()` | 直接映射 | 直接映射 | 需转换时间单位 |

### 6.2 Segment 映射

```python
# faster-whisper → Segment
Segment(
    start=float(seg.start),
    end=float(seg.end),
    text=seg.text.strip(),
    confidence=seg.avg_logprob,
)

# mlx-whisper → Segment
Segment(
    start=float(seg["start"]),
    end=float(seg["end"]),
    text=seg["text"].strip(),
    confidence=None,
)

# whisper.cpp → Segment
Segment(
    start=float(seg.t0) / 1000.0,  # ms → s
    end=float(seg.t1) / 1000.0,
    text=seg.text.strip(),
    confidence=None,
)
```

### 6.3 进度回调适配

- **faster-whisper**: 流式 generator，可在迭代时计算 `seg.end / total_duration`
- **mlx-whisper**: 不支持流式，只能完成后回调 1.0
- **whisper.cpp**: pywhispercpp 不支持进度回调，只能完成后回调

### 6.4 设备/计算类型选择

```python
def _auto_compute_type(hardware: HardwareCapabilities) -> str:
    """根据硬件自动选择最佳计算类型."""
    if hardware.device == "cuda":
        return "float16"
    elif hardware.device == "mps":
        return "float16"  # mlx-whisper
    return "int8"  # CPU
```

---

## 7. 依赖管理建议

### 7.1 pyproject.toml 可选依赖

```toml
[project.optional-dependencies]
whisper-faster = ["faster-whisper>=1.1"]
whisper-mlx = ["mlx-whisper>=0.4"]
whisper-cpp = ["pywhispercpp>=1.5"]
whisper-all = [
    "faster-whisper>=1.1",
    "mlx-whisper>=0.4; sys_platform == 'darwin'",
    "pywhispercpp>=1.5",
]
```

### 7.2 运行时动态导入

```python
def get_whisper_runner(backend: str, **kwargs) -> SttRunner:
    """工厂函数：根据后端名返回对应 runner."""
    if backend == "faster":
        from subtool.runners.faster_whisper import FasterWhisperRunner
        return FasterWhisperRunner(**kwargs)
    elif backend == "mlx":
        from subtool.runners.mlx_whisper import MlxWhisperRunner
        return MlxWhisperRunner(**kwargs)
    elif backend == "cpp":
        from subtool.runners.whisper_cpp import WhisperCppRunner
        return WhisperCppRunner(**kwargs)
    raise ValueError(f"未知 Whisper 后端: {backend}")
```

---

## 8. 最终建议

### 8.1 实施路线图

```
Phase 1 (立即): FasterWhisperRunner
    - 跨平台通用后端
    - 安装最简单
    - 覆盖所有用户

Phase 2 (短期): MlxWhisperRunner
    - Apple Silicon 优化
    - 与 Qwen3-ASR (MLX) 并列
    - macOS 用户体验最佳

Phase 3 (中期): WhisperCppRunner
    - 极致性能
    - 嵌入式/服务器场景
    - 量化模型支持
```

### 8.2 选择决策树

```
用户平台？
├── macOS Apple Silicon
│   ├── 追求极致性能 → mlx-whisper (Phase 2)
│   └── 需要离线/量化 → whisper.cpp (Phase 3)
├── Linux / Windows
│   ├── 有 NVIDIA GPU → faster-whisper (CUDA)
│   └── 仅 CPU → faster-whisper (int8)
└── 全平台分发 → faster-whisper (默认)
```

### 8.3 不推荐的方案及原因

| 方案 | 不推荐原因 |
|------|-----------|
| insanely-fast-whisper | 维护不活跃，仅 CLI，不适合集成 |
| whisperX | 架构重，依赖链复杂，词级对齐对中文有限 |
| faster-whisper-xxl | 与项目 orchestrator 功能重复 |
| whisper-medusa | 加速比有限，仅英语，生态小 |
| whisper-distillation | 学术为主，未产品化 |
| Parakeet/NeMo | 非 Whisper 架构，中文支持有限 |

---

## 附录: 数据来源

- GitHub API (2026-07-22)
- PyPI JSON API
- 各项目 README (raw.githubusercontent.com)
- 社区 benchmark 报告

---

*本报告由 Claude 自动生成，数据截至 2026-07-22。建议实施前再次确认各项目最新状态。*
