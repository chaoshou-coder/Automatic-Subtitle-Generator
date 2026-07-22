# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Local batch audio/video transcription tool (自动批量生成字幕工具) based on `faster-whisper`. Takes a file or directory of media and produces SRT/TXT transcripts using on-device inference. No server/API layer — pure CLI scripts.

## Commands

```bash
# Interactive entry point (recommended for end users)
python run.py

# Direct executor (no interaction, all params via CLI or .whisperrc)
python batch_transcribe.py <input_path> --lang zh --format srt,txt --fast --output-dir out_subtitles

# Preview without transcribing
python batch_transcribe.py <input_path> --dry-run --output-dir out_subtitles

# Pre-download model to models/ cache (~3GB)
python download_model.py

# Syntax check (the project's only "test" — no test framework exists)
python -m py_compile run.py batch_transcribe.py download_model.py
```

Requires Python 3.8+ and a system-installed `ffmpeg` on PATH. Dependencies (`faster-whisper`, `hf_transfer`) are auto-installed by `run.py` if missing.

## Architecture

Two-layer design with a strict separation of concerns:

**`run.py` — Interactive entry (user-facing orchestrator)**
- Dependency/FFmpeg detection and auto-install
- Interactive parameter collection (path, language, formats, output dir, dry-run, advanced)
- Optional model pre-download prompt
- Reads/writes `.whisperrc` for persisted defaults
- Delegates all real work to `batch_transcribe.run_task()`
- Chinese UI language; accepts Chinese/English language names via `LANG_ALIASES`

**`batch_transcribe.py` — Headless executor (stable API + CLI)**
- `run_task()` is the stable programmatic entry, called by both `run.py` and `main()`
- `main()` is the standalone CLI: merges `.whisperrc` config < CLI args, then calls `run_task()`
- Core pipeline: scan files → load model → transcribe per file (streaming segments) → atomic write
- `transcribe_file()` is a pure function: takes model + file + callback, returns (duration, elapsed). No I/O — testable.
- `SegmentWriter` is a callable that writes SRT/TXT from segment callbacks
- `ProgressTracker` computes running ETA across the batch

**`download_model.py` — Optional model pre-download**
- Downloads `large-v3` to `models/`; supports `hf_transfer` acceleration and HF mirror for CN environments

### Key design conventions

- **Atomic output**: writes to `*.part` temp files, then `os.replace()` to final path — interrupted runs never leave partial output
- **Skip-existing (resume)**: if *any* target output format exists for a source file, the file is skipped
- **Streaming transcription**: segments are written as they arrive via `on_segment` callback — avoids holding full audio/segment list in memory
- **Model resolution**: checks `models/` for a local `model.bin` (flat or `snapshots/` layout); falls back to downloading `large-v3`
- **Device auto-detection**: CUDA if `ctranslate2` reports GPU, else CPU; compute type float16 (CUDA) / int8 (CPU) with float32 fallback on CPU failure

### Configuration precedence

CLI args > `.whisperrc` (JSON in cwd) > built-in defaults. `.whisperrc` keys: `lang`, `output_formats`, `fast`, `skip_existing`, `output_dir`, `timestamp_in_txt`, `retries`, `beam_size`, `cpu_threads`, `gpu`.

## Platform Notes

- Windows UTF-8 console handling at top of `batch_transcribe.py` (`sys.platform == "win32"`: `chcp 65001`, stdout/stderr reconfigure)
- All paths use `pathlib.Path` (cross-platform)
- `signal.signal()` wrapped in try/except — fails safely on Windows
- `shutil.which("ffmpeg")` used for FFmpeg detection (cross-platform)
- No CI, no linter config, no test framework, no packaging (setup.py/pyproject.toml)
