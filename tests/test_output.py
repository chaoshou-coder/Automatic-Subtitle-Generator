"""Output 层测试 — SRT/TXT 写入和路径计算。"""

import pytest
from pathlib import Path

from subtool.core.protocol import Segment, TranscriptionResult
from subtool.output.writer import format_timestamp_srt, write_srt, write_txt
from subtool.output.paths import (
    SUPPORTED_EXTENSIONS,
    compute_output_paths,
    find_input_files,
    should_skip_outputs,
)


class TestTimestamp:
    def test_zero(self):
        assert format_timestamp_srt(0.0) == "00:00:00,000"

    def test_seconds(self):
        assert format_timestamp_srt(61.5) == "00:01:01,500"

    def test_hours(self):
        assert format_timestamp_srt(3661.250) == "01:01:01,250"

    def test_negative(self):
        assert format_timestamp_srt(-1.0) == "00:00:00,000"

    def test_nan(self):
        import math
        assert format_timestamp_srt(float("nan")) == "00:00:00,000"


class TestWriteSrt:
    def test_basic(self, tmp_path):
        result = TranscriptionResult(
            segments=[
                Segment(0.0, 1.5, "第一行"),
                Segment(1.5, 3.0, "第二行"),
            ],
            language="zh",
            duration=3.0,
        )
        out = tmp_path / "test.srt"
        write_srt(result, out)

        content = out.read_text(encoding="utf-8")
        assert "1\n00:00:00,000 --> 00:00:01,500\n第一行" in content
        assert "2\n00:00:01,500 --> 00:00:03,000\n第二行" in content

    def test_creates_parent_dirs(self, tmp_path):
        result = TranscriptionResult(
            segments=[Segment(0.0, 1.0, "test")],
            language=None,
            duration=1.0,
        )
        out = tmp_path / "sub" / "dir" / "test.srt"
        write_srt(result, out)
        assert out.exists()

    def test_empty_segments(self, tmp_path):
        result = TranscriptionResult(segments=[], language=None, duration=0.0)
        out = tmp_path / "empty.srt"
        write_srt(result, out)
        content = out.read_text(encoding="utf-8")
        assert content.strip() == ""


class TestWriteTxt:
    def test_basic(self, tmp_path):
        result = TranscriptionResult(
            segments=[
                Segment(0.0, 1.5, "第一行"),
                Segment(1.5, 3.0, "第二行"),
            ],
            language="zh",
            duration=3.0,
        )
        out = tmp_path / "test.txt"
        write_txt(result, out)
        content = out.read_text(encoding="utf-8")
        assert "第一行\n第二行" in content

    def test_with_timestamp(self, tmp_path):
        result = TranscriptionResult(
            segments=[Segment(0.0, 1.5, "你好")],
            language="zh",
            duration=1.5,
        )
        out = tmp_path / "test.txt"
        write_txt(result, out, timestamp=True)
        content = out.read_text(encoding="utf-8")
        assert "[0.00 -> 1.50]" in content
        assert "你好" in content


class TestAtomicWrite:
    def test_no_partial_file_on_success(self, tmp_path):
        """成功写入后不应有 .part 文件残留。"""
        result = TranscriptionResult(
            segments=[Segment(0.0, 1.0, "test")],
            language=None,
            duration=1.0,
        )
        out = tmp_path / "test.srt"
        write_srt(result, out)
        assert out.exists()
        assert not out.with_suffix(".srt.part").exists()


class TestPathCalculation:
    def test_compute_output_paths_no_dir(self, tmp_path):
        src = tmp_path / "video.mp4"
        src.touch()
        outputs = compute_output_paths(
            input_path=tmp_path,
            source_file=src,
            output_dir=None,
            output_formats=["srt", "txt"],
        )
        assert outputs["srt"] == tmp_path / "video.srt"
        assert outputs["txt"] == tmp_path / "video.txt"

    def test_compute_output_paths_with_dir(self, tmp_path):
        src = tmp_path / "sub" / "video.mp4"
        src.parent.mkdir(parents=True)
        src.touch()
        out_dir = tmp_path / "out"
        outputs = compute_output_paths(
            input_path=tmp_path,
            source_file=src,
            output_dir=out_dir,
            output_formats=["srt"],
        )
        assert outputs["srt"] == out_dir / "sub" / "video.srt"

    def test_should_skip(self, tmp_path):
        paths = {"srt": tmp_path / "a.srt", "txt": tmp_path / "a.txt"}
        assert not should_skip_outputs(paths)
        paths["srt"].touch()
        assert should_skip_outputs(paths)


class TestSupportedExtensions:
    def test_common_formats(self):
        assert ".mp4" in SUPPORTED_EXTENSIONS
        assert ".wav" in SUPPORTED_EXTENSIONS
        assert ".mkv" in SUPPORTED_EXTENSIONS
        assert ".mp3" in SUPPORTED_EXTENSIONS

    def test_unsupported(self):
        assert ".pdf" not in SUPPORTED_EXTENSIONS
        assert ".py" not in SUPPORTED_EXTENSIONS
