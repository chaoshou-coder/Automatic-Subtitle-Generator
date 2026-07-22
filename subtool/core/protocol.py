"""STT 运行时协议 — 所有转录后端实现的接口。"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, NamedTuple, Protocol, runtime_checkable


class Segment(NamedTuple):
    """单条转录片段。"""
    start: float
    end: float
    text: str
    confidence: float | None = None


class TranscriptionResult(NamedTuple):
    """单次转录的完整结果。"""
    segments: list[Segment]
    language: str | None
    duration: float


class HardwareCapabilities(NamedTuple):
    """当前设备的硬件能力。"""
    device: str        # "cpu" | "cuda" | "mps"
    memory_bytes: int


@runtime_checkable
class SttRunner(Protocol):
    """转录后端协议。每个后端实现此接口。"""

    @property
    def name(self) -> str:
        """后端显示名称，如 "qwen-asr"。"""
        ...

    def capabilities(self) -> HardwareCapabilities:
        """返回当前设备硬件能力。"""
        ...

    def can_run(self, source: Path, hardware: HardwareCapabilities) -> bool:
        """判断此 runner 能否在当前硬件上处理该文件。"""
        ...

    def transcribe(
        self,
        source: Path,
        *,
        language: str | None = None,
        on_progress: Callable[[float], None] | None = None,
    ) -> TranscriptionResult:
        """转录单个音频文件。

        Args:
            source: 预提取的音频文件路径（WAV/FLAC）。
            language: ISO 639-1 语言代码，None 表示自动检测。
            on_progress: 可选进度回调，参数为 0.0~1.0 的完成比例。

        Returns:
            标准格式的转录结果。

        Raises:
            TranscriptionError: 转录失败。
        """
        ...


class TranscriptionError(Exception):
    """转录过程中的错误基类。"""
    pass
