# subtool — 本地批量音视频字幕生成工具

基于 Qwen3-ASR 的离线批量转录工具。输入一个文件或目录，输出 SRT 字幕或 TXT 纯文本。全部在本地运行，无需网络。

## 特性

- **极速**：Apple Silicon M5 Pro 上 8.6 分钟视频仅需 8.8 秒（58 倍实时）
- **多模型**：支持 6 种 Qwen3-ASR 模型（0.6B/1.7B × 4bit/8bit/fp16）
- **标准输出**：SRT 字幕格式，可直接用于 FFmpeg 软/硬字幕
- **断点续传**：已处理文件自动跳过
- **跨平台**：Apple Silicon 原生支持，Linux/Windows 规划中
- **零配置**：默认模型即开即用

## 快速开始

### 安装

需要 Python 3.10+ 和 [FFmpeg](https://ffmpeg.org)。

```bash
# 克隆仓库
git clone https://github.com/chaoshou-coder/Automatic-Subtitle-Generator.git
cd Automatic-Subtitle-Generator

# 用 uv 创建环境并安装依赖
uv sync
```

### 下载模型

首次使用需要下载模型文件。默认使用 Qwen3-ASR-0.6B-4bit（~680MB）：

```bash
# 模型会自动下载到 ~/.cache/subtool/models/
# 也可以手动指定模型路径
```

### 运行

```bash
# 交互模式（引导式）
uv run subtool -i

# 命令行模式
uv run subtool path/to/video.mp4

# 指定输出目录和格式
uv run subtool path/to/video.mp4 --format srt,txt --output-dir ./subtitles

# 选择模型大小
uv run subtool path/to/video.mp4 --model-size 1.7B-8bit

# 预览模式（不执行）
uv run subtool path/to/video.mp4 --dry-run

# 批量处理目录
uv run subtool path/to/videos/ --format srt --output-dir ./subtitles
```

### 使用 pixi

```bash
pixi install
pixi run subtool path/to/video.mp4
```

## CLI 参数

```text
subtool [--version] [--lang LANG] [--format FORMAT]
        [--timestamp-in-txt] [--output-dir OUTPUT_DIR]
        [--no-skip-existing] [--backend BACKEND]
        [--model-size MODEL_SIZE] [--dry-run] [--interactive]
        [input_path]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `input_path` | 输入文件或目录 | （交互输入） |
| `--lang` | 语言代码 (zh/en/ja/...) 或 auto | auto |
| `--format` | 输出格式: srt 或 srt,txt | srt |
| `--timestamp-in-txt` | TXT 输出包含时间戳 | False |
| `--output-dir` | 输出目录 | 与源文件同级 |
| `--no-skip-existing` | 不跳过已有输出 | False |
| `--backend` | STT 后端 | qwen-asr |
| `--model-size` | 模型大小 | 0.6B-4bit |
| `--dry-run` | 仅预览不执行 | False |
| `--interactive, -i` | 交互模式 | False |

## 可用模型

| 模型 | 大小 | 速度 (M5 Pro) | 推荐场景 |
|------|------|---------------|----------|
| **0.6B-4bit** ⭐ | 680 MB | 8.8s (58x) | 日常首选 |
| 0.6B-8bit | 964 MB | 9.2s (56x) | 质量略好 |
| 0.6B | 1.8 GB | 13.0s (40x) | 全精度小模型 |
| 1.7B-4bit | 1.5 GB | 12.5s (41x) | 平衡选择 |
| 1.7B-8bit | 2.3 GB | 18.0s (29x) | 高质量 |
| 1.7B | 4.4 GB | 28.5s (18x) | 最佳质量 |

## 项目结构

```
subtool/
├── cli/main.py          # CLI 入口（交互 + 命令行）
├── core/
│   ├── protocol.py      # SttRunner Protocol 接口
│   └── orchestrator.py  # 批量编排器
├── runners/
│   └── mlx_qwen3.py     # Qwen3 ASR (MLX) 后端
├── media/extract.py     # ffmpeg 音频提取
├── output/
│   ├── writer.py        # SRT/TXT 原子写入
│   └── paths.py         # 文件扫描 + 路径计算
└── hardware/detect.py   # CPU/CUDA/Metal 设备检测
```

## 系统要求

- Python 3.10+
- FFmpeg（命令行可用）
- macOS ARM64 (Apple Silicon) — 完全支持
- Linux / Windows — 规划中

## 开发

```bash
# 运行测试
uv run pytest tests/ -v

# 性能基准测试
uv run python benchmark.py
```

## License

MIT
