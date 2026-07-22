"""Protocol 和数据结构测试。"""

import pytest
from pathlib import Path

from subtool.core.protocol import (
    HardwareCapabilities,
    Segment,
    SttRunner,
    TranscriptionError,
    TranscriptionResult,
)


class TestSegment:
    def test_basic(self):
        seg = Segment(start=0.0, end=1.5, text="你好")
        assert seg.start == 0.0
        assert seg.end == 1.5
        assert seg.text == "你好"
        assert seg.confidence is None

    def test_with_confidence(self):
        seg = Segment(start=0.0, end=1.5, text="你好", confidence=0.95)
        assert seg.confidence == 0.95

    def test_immutable(self):
        seg = Segment(start=0.0, end=1.5, text="你好")
        with pytest.raises(AttributeError):
            seg.start = 1.0


class TestTranscriptionResult:
    def test_basic(self):
        result = TranscriptionResult(
            segments=[Segment(0.0, 1.0, "你好")],
            language="zh",
            duration=1.0,
        )
        assert len(result.segments) == 1
        assert result.language == "zh"
        assert result.duration == 1.0


class TestHardwareCapabilities:
    def test_basic(self):
        hw = HardwareCapabilities(device="mps", memory_bytes=48 * 1024**3)
        assert hw.device == "mps"
        assert hw.memory_bytes == 48 * 1024**3


class TestProtocolCheck:
    def test_runtime_checkable(self):
        """验证 Protocol 可以用 isinstance 检查。"""

        class FakeRunner:
            name = "fake"
            def capabilities(self):
                return HardwareCapabilities("cpu", 8 * 1024**3)
            def can_run(self, source, hardware):
                return True
            def transcribe(self, source, language=None, on_progress=None):
                return TranscriptionResult([], None, 0.0)

        runner = FakeRunner()
        assert isinstance(runner, SttRunner)

    def test_non_compliant(self):
        """不实现全部方法的对象不是 SttRunner。"""
        class BadRunner:
            name = "bad"

        assert not isinstance(BadRunner(), SttRunner)
