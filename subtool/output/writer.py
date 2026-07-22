"""输出写入 — SRT/TXT 格式化和原子写入。"""

from __future__ import annotations

import math
from pathlib import Path

from subtool.core.protocol import Segment, TranscriptionResult


def format_timestamp_srt(seconds: float) -> str:
    """将秒数格式化为 SRT 时间戳：HH:MM:SS,mmm。"""
    try:
        seconds_f = float(seconds)
    except Exception:
        seconds_f = 0.0

    if math.isnan(seconds_f):
        seconds_f = 0.0

    if seconds_f < 0:
        total_ms = int(seconds_f * 1000 - 0.5)
    else:
        total_ms = int(seconds_f * 1000 + 0.5)
    if total_ms < 0:
        total_ms = 0

    h, rem_ms = divmod(total_ms, 3_600_000)
    m, rem_ms = divmod(rem_ms, 60_000)
    s, ms = divmod(rem_ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(result: TranscriptionResult, path: Path) -> None:
    """将转录结果写入 SRT 文件（原子写入）。"""
    lines: list[str] = []
    for idx, seg in enumerate(result.segments, 1):
        lines.append(str(idx))
        lines.append(f"{format_timestamp_srt(seg.start)} --> {format_timestamp_srt(seg.end)}")
        lines.append(seg.text.strip())
        lines.append("")

    _atomic_write(path, "\n".join(lines))


def write_txt(
    result: TranscriptionResult,
    path: Path,
    *,
    timestamp: bool = False,
) -> None:
    """将转录结果写入 TXT 文件（原子写入）。"""
    lines: list[str] = []
    for seg in result.segments:
        if timestamp:
            lines.append(f"[{seg.start:.2f} -> {seg.end:.2f}] {seg.text.strip()}")
        else:
            lines.append(seg.text.strip())

    _atomic_write(path, "\n".join(lines) + "\n")


def _atomic_write(final_path: Path, content: str) -> None:
    """原子写入：先写 .part 临时文件，再 rename 为最终文件。"""
    tmp_path = final_path.with_suffix(final_path.suffix + ".part")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(final_path)
