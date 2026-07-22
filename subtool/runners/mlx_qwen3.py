"""Qwen3 ASR runner — 基于 mlx-audio (Apple Silicon 优化)。

使用 MLX 格式的 Qwen3-ASR 模型，通过 mlx-audio CLI 子进程调用。
避免了 mlx-audio 库的循环导入 bug，同时获得 MLX 在 Apple Silicon 上的
原生性能。

模型路径可配置，默认 ~/LLM/models/Qwen3-ASR-1.7B-8bit。
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
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


DEFAULT_MODEL = Path.home() / "LLM" / "models" / "Qwen3-ASR-1.7B-8bit"


class MlxQwen3Runner:
    """基于 mlx-audio CLI 的 Qwen3 ASR runner。"""

    def __init__(self, model_path: Path | None = None) -> None:
        self._model_path = model_path or DEFAULT_MODEL

    @property
    def name(self) -> str:
        return "qwen3-asr-mlx"

    def capabilities(self) -> HardwareCapabilities:
        return detect_hardware()

    def can_run(self, source: Path, hardware: HardwareCapabilities) -> bool:
        return hardware.memory_bytes >= 4 * 1024 * 1024 * 1024

    def transcribe(
        self,
        source: Path,
        *,
        language: str | None = None,
        on_progress: Callable[[float], None] | None = None,
    ) -> TranscriptionResult:
        """转录单个音频文件。"""
        if not self._model_path.exists():
            raise TranscriptionError(
                f"模型不存在: {self._model_path}\n"
                f"请下载 Qwen3-ASR MLX 模型到该路径。"
            )

        # 创建临时输出文件
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as tmp:
            tmp_stem = tmp.name[:-5]  # 去掉 .json 后缀

        try:
            # 调用 mlx-audio CLI（使用当前 Python 解释器，确保在 venv 中）
            import sys
            cmd = [
                sys.executable,
                "-m", "mlx_audio.stt.generate",
                "--model", str(self._model_path),
                "--audio", str(source),
                "--output-path", tmp_stem,
                "--format", "json",
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 分钟超时
            )

            if result.returncode != 0:
                raise TranscriptionError(
                    f"mlx-audio 转录失败: {result.stderr[-300:]}"
                )

            # 读取 JSON 输出
            json_path = Path(tmp_stem + ".json")
            if not json_path.exists():
                raise TranscriptionError("mlx-audio 未生成输出文件")

            data = json.loads(json_path.read_text())

            # 解析 segments
            segments: list[Segment] = []
            for seg in data.get("segments", []):
                segments.append(Segment(
                    start=float(seg.get("start", 0.0)),
                    end=float(seg.get("end", 0.0)),
                    text=seg.get("text", "").strip(),
                    confidence=None,
                ))

            duration = segments[-1].end if segments else 0.0

            return TranscriptionResult(
                segments=segments,
                language=language,
                duration=duration,
            )

        except subprocess.TimeoutExpired:
            raise TranscriptionError("转录超时（>10分钟）")
        except Exception as e:
            if isinstance(e, TranscriptionError):
                raise
            raise TranscriptionError(f"转录失败: {e}") from e
        finally:
            # 清理临时文件
            for suffix in [".json", ".txt", ".srt"]:
                try:
                    p = Path(tmp_stem + suffix)
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass
