"""音频提取 — 通过 ffmpeg 将视频/音频转为 WAV。"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


class FFmpegError(Exception):
    """ffmpeg 执行错误。"""
    pass


class NoAudioTrackError(FFmpegError):
    """媒体文件没有音频轨道。"""
    pass


def check_ffmpeg() -> bool:
    """检查 ffmpeg 是否可用。"""
    return shutil.which("ffmpeg") is not None


def extract_audio(source: Path, output: Path | None = None) -> Path:
    """从媒体文件提取音频为 16-bit PCM WAV (16kHz, mono)。

    Args:
        source: 输入媒体文件路径。
        output: 输出 WAV 路径，None 则创建临时文件。

    Returns:
        输出 WAV 文件路径。

    Raises:
        FFmpegError: ffmpeg 执行失败。
        NoAudioTrackError: 文件没有音频轨道。
    """
    if not check_ffmpeg():
        raise FFmpegError("ffmpeg not found on PATH")

    if output is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output = Path(tmp.name)
        tmp.close()

    cmd = [
        "ffmpeg", "-y",
        "-i", str(source),
        "-vn",                    # 无视频
        "-acodec", "pcm_s16le",   # 16-bit PCM
        "-ar", "16000",           # 16kHz
        "-ac", "1",               # 单声道
        str(output),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        stderr = result.stderr.lower()
        if "stream matches no streams" in stderr or "audio" in stderr:
            raise NoAudioTrackError(f"No audio track in {source.name}")
        raise FFmpegError(
            f"ffmpeg failed on {source.name}: {result.stderr[-200:]}"
        )

    return output


def probe_audio(source: Path) -> dict[str, str] | None:
    """探测媒体文件的音频流信息。"""
    if not check_ffmpeg():
        return None

    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-select_streams", "a",
        str(source),
    ]

    try:
        import json
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        if streams:
            return {
                "codec": streams[0].get("codec_name", "unknown"),
                "sample_rate": streams[0].get("sample_rate", "unknown"),
                "channels": str(streams[0].get("channels", "unknown")),
                "duration": streams[0].get("duration", "unknown"),
            }
    except Exception:
        pass

    return None
