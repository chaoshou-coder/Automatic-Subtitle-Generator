# Faster Whisper 批量转录工具

基于 `faster-whisper` 的本地批量音视频转录脚本，面向普通用户的入口是 `run.py`：自动校验/安装依赖、检测 FFmpeg、可选预下载模型，并交互收集参数后调用执行器批处理。

## 文档

- 详细手册：[PROJECT_MANUAL.md](docs/PROJECT_MANUAL.md)
- 开发者说明：[DEVELOPMENT.md](docs/DEVELOPMENT.md)
- 变更记录：[CHANGELOG.md](CHANGELOG.md)

## 快速开始

### 1) 安装 Python

需要 Python 3.8+（包含 pip）。

### 2) 安装 FFmpeg

请确保 `ffmpeg -version` 可用。安装教程见：[FFMPEG_INSTALL.md](docs/FFMPEG_INSTALL.md)

### 3) 一键运行（推荐）

```bash
python run.py
```

## 常用参数（执行器）

`run.py` 会在最后调用执行器 `batch_transcribe.py`。高级用户也可以直接运行执行器（无交互）：

```text
python batch_transcribe.py <input_path> [--lang zh|auto] [--format srt|srt,txt]
                           [--fast] [--beam-size N] [--gpu N] [--cpu-threads N]
                           [--timestamp-in-txt] [--retries N]
                           [--output-dir <dir>] [--dry-run] [--skip-existing|--no-skip-existing]
                           [--config <path>]
```

提示：
- `--lang auto` 表示自动检测语言
- 强烈建议配合 `--output-dir` 使用，避免在源文件目录产生输出文件

## 主要特性

- 一键引导：依赖校验/自动安装 + 交互配置 + 批处理执行
- 断点续传：任一目标输出存在即跳过该文件（默认开启，可用 `--no-skip-existing` 覆盖）
- 输出格式精简：默认 `srt`，可选 `txt`
- 自动设备选择：有 CUDA 用 CUDA，没有则用 CPU
- 性能开关：`--fast` 一键将 `beam_size` 降到 1（通常 3–5x 加速）
- 多卡支持：`--gpu N` 选择使用第 N 张卡
- 输出目录：`--output-dir` 将字幕集中到独立目录，且保留原目录结构
- 预览模式：`--dry-run` 先列出将处理的文件与输出路径，避免误操作

## 输出规则

- 输出写在源文件同级目录：`video.mp4 -> video.srt`（可选 `video.txt`）
- 临时文件：转录过程中写入 `*.srt.part / *.txt.part`，成功后原子替换为最终文件（避免中途失败留下半文件）

说明：
- `srt` 是字幕格式，可直接用于 FFmpeg 软/硬字幕。
- `txt` 是纯文本转写，适合阅读/检索，不是字幕格式。

## 示例（执行器）

```bash
# 先预览将处理哪些文件，并把输出写到 out_subtitles/
python batch_transcribe.py "C:\Videos" --lang auto --format srt,txt --dry-run --output-dir out_subtitles

# 确认无误后执行（默认跳过已有输出）
python batch_transcribe.py "C:\Videos" --lang auto --format srt,txt --output-dir out_subtitles

# 如需覆盖重生成
python batch_transcribe.py "C:\Videos" --lang auto --format srt --output-dir out_subtitles --no-skip-existing
```

## 配置文件（.whisperrc）

在运行目录放置 `.whisperrc`（JSON），执行器会自动读取作为默认参数（命令行参数优先生效）：

```json
{
  "lang": "zh",
  "output_formats": ["srt", "txt"],
  "fast": true,
  "skip_existing": true,
  "output_dir": "out_subtitles"
}
```

优先级规则：
- 命令行参数 > `.whisperrc` 配置 > 内置默认值
