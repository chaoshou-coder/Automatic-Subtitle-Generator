"""输出路径计算和文件过滤。"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

SUPPORTED_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".mp4", ".mkv",
    ".flac", ".mov", ".avi", ".webm",
    ".ogg", ".opus", ".wma",
}

OUTPUT_FORMATS = {"srt", "txt"}


def find_input_files(input_path: Path) -> list[Path]:
    """扫描输入路径，返回所有支持的媒体文件。"""
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in SUPPORTED_EXTENSIONS else []

    files: list[Path] = []
    for p in input_path.rglob("*"):
        try:
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(p)
        except Exception:
            continue
    return sorted(files)


def compute_output_paths(
    *,
    input_path: Path,
    source_file: Path,
    output_dir: Path | None,
    output_formats: Sequence[str],
) -> dict[str, Path]:
    """计算某个源文件对应的输出路径。"""
    input_root = input_path if input_path.is_dir() else input_path.parent
    try:
        rel = source_file.relative_to(input_root)
    except Exception:
        rel = source_file.name

    base = (output_dir / rel) if output_dir is not None else source_file
    return {
        fmt: base.with_suffix(f".{fmt}")
        for fmt in output_formats
        if fmt in OUTPUT_FORMATS
    }


def should_skip_outputs(output_paths: dict[str, Path]) -> bool:
    """如果任一目标输出已存在，跳过该文件。"""
    return any(p.exists() for p in output_paths.values())
