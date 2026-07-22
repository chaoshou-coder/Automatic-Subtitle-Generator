"""Hardware 层测试 — 设备检测。"""

from subtool.hardware.detect import (
    default_compute_type,
    detect_hardware,
)
from subtool.core.protocol import HardwareCapabilities


class TestDetectHardware:
    def test_returns_capabilities(self):
        hw = detect_hardware()
        assert isinstance(hw, HardwareCapabilities)
        assert hw.device in ("cpu", "cuda", "mps")
        assert hw.memory_bytes > 0

    def test_memory_reasonable(self):
        hw = detect_hardware()
        # 内存在 1GB 到 1TB 之间
        assert 1 * 1024**3 < hw.memory_bytes < 1024 * 1024**3

    def test_this_mac_is_mps(self):
        """这台 M5 Pro Mac 应该检测到 MPS。"""
        import sys
        if sys.platform == "darwin":
            hw = detect_hardware()
            assert hw.device == "mps"


class TestComputeType:
    def test_cuda_float16(self):
        assert default_compute_type("cuda") == "float16"

    def test_cpu_int8(self):
        assert default_compute_type("cpu") == "int8"

    def test_mps_int8(self):
        assert default_compute_type("mps") == "int8"
