"""编排器 — 连接 media → runner → output 的主循环。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from subtool.core.protocol import (
    SttRunner,
    TranscriptionError,
    TranscriptionResult,
)
from subtool.hardware.detect import detect_hardware
from subtool.media.extract import extract_audio
from subtool.output.paths import (
    compute_output_paths,
    find_input_files,
    should_skip_outputs,
)
from subtool.output.writer import write_srt, write_txt


@dataclass
class FileResult:
    """单个文件的处理结果。"""
    source: Path
    status: str          # "success" | "skipped" | "failed"
    message: str = ""
    duration: float = 0.0   # 音频时长（秒）
    wall_seconds: float = 0.0


@dataclass
class BatchResult:
    """批量处理结果。"""
    files: list[FileResult] = field(default_factory=list)

    @property
    def success_count(self) -> int:
        return sum(1 for f in self.files if f.status == "success")

    @property
    def skip_count(self) -> int:
        return sum(1 for f in self.files if f.status == "skipped")

    @property
    def fail_count(self) -> int:
        return sum(1 for f in self.files if f.status == "failed")


def run_batch(
    *,
    input_path: Path,
    output_dir: Path | None,
    output_formats: Sequence[str],
    runner: SttRunner,
    language: str | None = None,
    skip_existing: bool = True,
    timestamp_in_txt: bool = False,
    on_progress: Callable[[str, float], None] | None = None,
    on_file_done: Callable[[FileResult], None] | None = None,
    stop_check: Callable[[], bool] | None = None,
) -> BatchResult:
    """执行批量转录。

    Args:
        input_path: 输入文件或目录。
        output_dir: 输出目录，None 则与源文件同级。
        output_formats: 输出格式列表。
        runner: STT 后端实例。
        language: 语言代码，None 表示自动检测。
        skip_existing: 是否跳过已有输出。
        timestamp_in_txt: TXT 是否包含时间戳。
        on_progress: 进度回调 (文件名, 进度0-1)。
        on_file_done: 单文件完成回调。
        stop_check: 检查是否应停止（返回 True 则停止）。

    Returns:
        批量处理结果。
    """
    hardware = detect_hardware()
    files = find_input_files(input_path)

    resolved_output_dir: Path | None = None
    if output_dir is not None:
        resolved_output_dir = output_dir.expanduser()
        try:
            resolved_output_dir = resolved_output_dir.resolve()
        except Exception:
            pass

    result = BatchResult()

    for file_path in files:
        if stop_check is not None and stop_check():
            break

        output_paths = compute_output_paths(
            input_path=input_path,
            source_file=file_path,
            output_dir=resolved_output_dir,
            output_formats=output_formats,
        )

        if skip_existing and should_skip_outputs(output_paths):
            fr = FileResult(
                source=file_path,
                status="skipped",
                message="输出已存在",
            )
            result.files.append(fr)
            if on_file_done:
                on_file_done(fr)
            continue

        try:
            fr = _process_single_file(
                file_path=file_path,
                output_paths=output_paths,
                runner=runner,
                language=language,
                timestamp_in_txt=timestamp_in_txt,
                on_progress=on_progress,
                hardware=hardware,
            )
        except TranscriptionError as e:
            fr = FileResult(
                source=file_path,
                status="failed",
                message=str(e),
            )
        except Exception as e:
            fr = FileResult(
                source=file_path,
                status="failed",
                message=f"{type(e).__name__}: {e}",
            )

        result.files.append(fr)
        if on_file_done:
            on_file_done(fr)

    return result


def _process_single_file(
    *,
    file_path: Path,
    output_paths: dict[str, Path],
    runner: SttRunner,
    language: str | None,
    timestamp_in_txt: bool,
    on_progress: Callable[[str, float], None] | None,
    hardware,
) -> FileResult:
    """处理单个文件：提取音频 → 转录 → 写入输出。"""
    import time

    started = time.time()

    # 检查 runner 能否处理
    if not runner.can_run(file_path, hardware):
        return FileResult(
            source=file_path,
            status="failed",
            message=f"{runner.name} 无法在当前硬件上运行（内存不足）",
        )

    # 提取音频
    audio_path = extract_audio(file_path)

    try:
        # 转录
        def progress_cb(p: float) -> None:
            if on_progress:
                on_progress(file_path.name, p)

        transcription = runner.transcribe(
            audio_path,
            language=language,
            on_progress=progress_cb if on_progress else None,
        )

        # 写入输出
        for fmt, out_path in output_paths.items():
            if fmt == "srt":
                write_srt(transcription, out_path)
            elif fmt == "txt":
                write_txt(transcription, out_path, timestamp=timestamp_in_txt)
    finally:
        # 清理临时音频文件
        try:
            audio_path.unlink()
        except Exception:
            pass

    return FileResult(
        source=file_path,
        status="success",
        duration=transcription.duration,
        wall_seconds=time.time() - started,
    )
