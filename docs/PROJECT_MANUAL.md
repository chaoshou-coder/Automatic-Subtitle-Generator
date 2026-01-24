# Faster Whisper 批量转录工具 - 项目手册

## 1. 这是什么

这是一个基于 `faster-whisper` 的本地批量转录脚本，目标是“少参数、能跑、好续跑”：
- 默认入口为 `run.py`（面向人类使用，负责交互与环境准备）
- 默认语言 `zh`、默认输出 `srt`
- 默认不重试（避免对不可恢复错误浪费时间）
- 自动选择 CUDA/CPU（用户不必理解 `--device/--compute-type`）

## 2. 安装与运行环境

### 2.1 Python 依赖

```bash
pip install faster-whisper
```

也可以直接运行 `python run.py`，脚本会自动检测并安装缺失依赖。

### 2.2 FFmpeg

脚本需要 FFmpeg 解码音视频。
请先安装 FFmpeg，并确保 `ffmpeg -version` 可用。安装教程见：[FFMPEG_INSTALL.md](FFMPEG_INSTALL.md)

### 2.3 模型（可选预下载）

脚本优先从 `models/` 目录加载模型；如果不存在，会按 `faster-whisper` 规则自动下载。`models/` 为本地缓存目录，不应提交到 Git 仓库。

如需预下载：
```bash
python download_model.py
```

说明：
- `download_model.py` 可能下载出 HuggingFace 的 `snapshots/.../model.bin` 结构；
- `batch_transcribe.py` 会在 `models/` 目录下递归查找 `model.bin` 并自动识别正确的模型目录。

## 3. 使用方式

### 3.1 一键运行（推荐）

```bash
python run.py
```

运行脚本会完成：
1) 校验/自动安装 Python 依赖（如 `faster-whisper`）  
2) 校验 FFmpeg 是否可用（不可用时给出安装提示）  
3) 可选预下载模型到 `models/`  
4) 交互选择参数后开始批处理  

交互流程（仅问必要问题）：
1) 输入文件/文件夹路径  
2) 选择语言（默认 `zh`，可输入 `auto`）  
3) 输出格式（默认 `srt`，可选 `srt,txt`）  
4) 若选择 `txt`，可选是否写入时间戳  
5) 可选输出目录（推荐）与 dry-run 预览（推荐）

### 3.2 执行器模式（高级用户）

`batch_transcribe.py` 是纯执行器：不包含交互逻辑，只接受命令行参数并直接开始处理。

```bash
python batch_transcribe.py "C:\Videos" --lang zh --format srt,txt --fast
```

常用增强功能：
- 预览（不执行）：`--dry-run`
- 输出到独立目录：`--output-dir out_subtitles`
- 覆盖/跳过控制：默认跳过已有输出；如需覆盖重新生成，使用 `--no-skip-existing`
- 配置文件：默认读取运行目录下的 `.whisperrc`（也可用 `--config path` 指定）

## 4. 参数参考（CLI）

| 参数 | 作用 | 默认值 |
| :--- | :--- | :--- |
| `input_path` | 输入文件或目录 | (交互输入) |
| `--lang` | `zh/en/yue/...` 或 `auto` | `zh` |
| `--format` | `srt` 或 `srt,txt` | `srt` |
| `--timestamp-in-txt` | TXT 带时间戳 | `False` |
| `--retries` | 失败重试次数 | `0` |
| `--fast` | 快速模式（等价默认 `--beam-size 1`） | `False` |
| `--beam-size` | 解码 beam_size（质量 vs 速度） | `5` |
| `--cpu-threads` | CPU 线程数 | `os.cpu_count()` |
| `--gpu` | 选择第 N 张 GPU（仅 CUDA 可用时生效） | `0` |
| `--dry-run` | 仅预览将处理哪些文件与输出位置 | `False` |
| `--output-dir` | 输出目录（保留相对目录结构） | (同级输出) |
| `--skip-existing` | 跳过已有输出的文件 | `True` |
| `--no-skip-existing` | 覆盖重新生成 | `False` |
| `--config` | 配置文件路径 | (自动读取 .whisperrc) |

## 5. 输出与断点续传

### 5.1 输出文件

- 默认输出与源文件同目录同名，仅扩展名改变：`a.mp4 -> a.srt`（可选 `a.txt`）
- 转录过程中写到临时文件：`a.srt.part / a.txt.part`  
  成功后用 `os.replace()` 原子替换为最终文件，避免中断留下半文件。

补充说明：
- `srt` 是字幕格式，可直接用于 FFmpeg 软/硬字幕。
- `txt` 是纯文本转写，不是字幕格式，不能直接用于烧录。

### 5.2 输出目录（--output-dir）

当指定 `--output-dir` 时，会在输出目录内保留输入目录的相对结构，例如：
- 输入：`in/a/b.mp4`
- 输出：`out/a/b.srt`

### 5.3 配置文件（.whisperrc）

在运行目录创建 `.whisperrc`（JSON）可以保存常用参数，示例：

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

### 5.4 断点续传策略（重要）

只要“任一目标格式输出已存在”，就跳过该源文件：
- 例：第一次只输出 `srt`，第二次改为 `srt,txt`，如果 `srt` 已存在则会跳过，不会重转覆盖已有 `srt`。

如果你希望强制重跑，手动删除对应输出文件即可。

## 6. 性能调优建议

### 6.1 快速优先（大多数用户）

推荐直接加 `--fast`（beam_size=1）：
```bash
python batch_transcribe.py "C:\Videos" --lang zh --format srt --fast
```

### 6.2 GPU 利用率

- 多卡：`--gpu N` 选择具体显卡。

### 6.3 CPU 线程

默认取 `os.cpu_count()`；如遇到 CPU 抢占导致卡顿，可手动降低：
```bash
python batch_transcribe.py "C:\Videos" --cpu-threads 8
```

## 7. 常见问题（FAQ）

**Q: 为什么 GPU 没有被使用？**  
A: 请确认安装的 CTranslate2 支持 CUDA，且系统 CUDA 驱动环境正常。脚本会自动检测，有 CUDA 即用 CUDA。

**Q: 如何把字幕合到视频？**  
A: 使用 FFmpeg：  
- 软字幕：`ffmpeg -i video.mp4 -i video.srt -c copy -c:s mov_text output.mp4`  
- 硬字幕：`ffmpeg -i video.mp4 -vf subtitles=video.srt output.mp4`
