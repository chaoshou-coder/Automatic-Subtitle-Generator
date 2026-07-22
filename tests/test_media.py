"""Media 层测试 — ffmpeg 音频提取。"""

import subprocess
from pathlib import Path

import pytest

from subtool.media.extract import (
    NoAudioTrackError,
    check_ffmpeg,
    extract_audio,
    probe_audio,
)


@pytest.fixture
def test_wav(tmp_path):
    """生成一个 2 秒 440Hz 正弦波测试音频。"""
    path = tmp_path / "tone.wav"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-ar", "16000", "-ac", "-1",
            "-c:a", "pcm_s16le",
            str(path),
        ],
        capture_output=True,
        check=True,
    )
    return path


@pytest.fixture
def test_video(tmp_path):
    """生成一个 2 秒测试视频（带音频）。"""
    path = tmp_path / "test.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=blue:s=320x240:d=2",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-c:v", "libx264", "-c:a", "aac",
            "-shortest",
            str(path),
        ],
        capture_output=True,
        check=True,
    )
    return path


class TestFFmpegCheck:
    def test_available(self):
        assert check_ffmpeg() is True


class TestExtractAudio:
    def test_from_wav(self, test_wav, tmp_path):
        out = tmp_path / "output.wav"
        result = extract_audio(test_wav, out)
        assert result.exists()
        assert result.stat().st_size > 0

    def test_from_video(self, test_video, tmp_path):
        out = tmp_path / "extracted.wav"
        result = extract_audio(test_video, out)
        assert result.exists()
        assert result.stat().st_size > 0

    def test_auto_temp_file(self, test_wav):
        result = extract_audio(test_wav)
        assert result.exists()
        assert result.stat().st_size > 0
        result.unlink()  # 清理临时文件

    def test_output_is_16khz_mono(self, test_wav, tmp_path):
        import wave
        out = tmp_path / "out.wav"
        extract_audio(test_wav, out)
        with wave.open(str(out), "rb") as wf:
            assert wf.getframerate() == 16000
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2  # 16-bit


class TestProbeAudio:
    def test_probe_wav(self, test_wav):
        info = probe_audio(test_wav)
        assert info is not None
        assert "sample_rate" in info

    def test_probe_video(self, test_video):
        info = probe_audio(test_video)
        assert info is not None
