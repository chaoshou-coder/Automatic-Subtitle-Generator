"""Qwen3 ASR 性能基准测试 — 对比不同参数量和量化版本。"""

import json
import subprocess
import sys
import time
from pathlib import Path


AUDIO = Path("jonathan_audio.wav")
MODELS = [
    ("0.6B-4bit",  "~/LLM/asr/Qwen3-ASR-0.6B-4bit"),
    ("0.6B-8bit",  "~/LLM/asr/Qwen3-ASR-0.6B-8bit"),
    ("0.6B-fp16",  "~/LLM/asr/Qwen3-ASR-0.6B"),
    ("1.7B-4bit",  "~/LLM/asr/Qwen3-ASR-1.7B-4bit"),
    ("1.7B-8bit",  "~/LLM/asr/Qwen3-ASR-1.7B-8bit"),
    ("1.7B-fp16",  "~/LLM/asr/Qwen3-ASR-1.7B"),
]


def run_transcription(model_path: str) -> dict:
    """运行一次转录，返回性能数据。"""
    model_path = str(Path(model_path).expanduser())
    cmd = [
        sys.executable,
        "-m", "mlx_audio.stt.generate",
        "--model", model_path,
        "--audio", str(AUDIO),
        "--output-path", "/tmp/bench_out",
        "--format", "json",
    ]

    # 预热（第一次加载模型到内存）
    # 不计入时间

    # 正式计时
    start = time.perf_counter()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    elapsed = time.perf_counter() - start

    # 读取结果
    segments = 0
    json_path = Path("/tmp/bench_out.json")
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text())
            segments = len(data.get("segments", []))
        except Exception:
            pass

    return {
        "elapsed_s": round(elapsed, 2),
        "segments": segments,
        "success": result.returncode == 0,
        "error": result.stderr[-200:] if result.returncode != 0 else None,
    }


def main():
    print("=" * 60)
    print("Qwen3 ASR 性能基准测试")
    print(f"音频: {AUDIO.name} (8.6 分钟)")
    print(f"设备: Apple M5 Pro, 48 GB")
    print("=" * 60)

    results = []
    for name, path in MODELS:
        model_dir = Path(path).expanduser()
        if not model_dir.exists():
            print(f"\n⏭  {name}: 模型不存在，跳过")
            continue

        print(f"\n🔄 测试 {name}...", flush=True)
        try:
            data = run_transcription(path)
            data["model"] = name
            results.append(data)

            if data["success"]:
                rtf = data["elapsed_s"] / 516  # 音频时长 516s
                print(f"  ✅ {data['elapsed_s']:.1f}s | "
                      f"{data['segments']} segments | "
                      f"RTF: {rtf:.3f}x")
            else:
                print(f"  ❌ 失败: {data['error'][:100]}")
        except subprocess.TimeoutExpired:
            print(f"  ⏰ 超时")
            results.append({"model": name, "success": False, "error": "timeout"})
        except Exception as e:
            print(f"  ❌ 错误: {e}")
            results.append({"model": name, "success": False, "error": str(e)})

    # 汇总
    print("\n" + "=" * 60)
    print("汇总")
    print("=" * 60)
    print(f"{'模型':<15} {'耗时(s)':<10} {'Segments':<10} {'RTF':<10} {'状态'}")
    print("-" * 60)
    for r in results:
        if r.get("success"):
            rtf = r["elapsed_s"] / 516
            print(f"{r['model']:<15} {r['elapsed_s']:<10.1f} {r['segments']:<10} {rtf:<10.3f} ✅")
        else:
            print(f"{r['model']:<15} {'—':<10} {'—':<10} {'—':<10} ❌")


if __name__ == "__main__":
    main()
