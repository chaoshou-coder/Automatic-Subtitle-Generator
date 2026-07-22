"""硬件设备检测 — CPU/CUDA/Metal。"""

from __future__ import annotations

import os

from subtool.core.protocol import HardwareCapabilities


def detect_hardware() -> HardwareCapabilities:
    """检测当前设备的硬件能力。

    优先级：CUDA > Metal (Apple Silicon) > CPU。
    """
    # 检查 CUDA
    try:
        import torch
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            mem = torch.cuda.get_device_properties(idx).total_memory
            return HardwareCapabilities(device="cuda", memory_bytes=int(mem))
    except ImportError:
        pass

    # 检查 Metal (Apple Silicon)
    try:
        import torch
        if torch.backends.mps.is_available():
            # MPS 使用统一内存，报告系统总内存
            import sys
            if sys.platform == "darwin":
                import subprocess
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True, text=True, check=True,
                )
                mem = int(result.stdout.strip())
                return HardwareCapabilities(device="mps", memory_bytes=mem)
    except (ImportError, Exception):
        pass

    # 回退到 CPU
    mem = _cpu_memory()
    return HardwareCapabilities(device="cpu", memory_bytes=mem)


def _cpu_memory() -> int:
    """获取系统总内存字节数。"""
    try:
        import sys
        if sys.platform == "darwin":
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, check=True,
            )
            return int(result.stdout.strip())
        elif sys.platform == "linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 8 * 1024 * 1024 * 1024  # 默认 8 GB


def default_compute_type(device: str) -> str:
    """根据设备选择最佳计算精度。"""
    if device == "cuda":
        return "float16"
    return "int8"
