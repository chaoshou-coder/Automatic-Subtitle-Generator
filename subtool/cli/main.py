"""CLI 入口 — 交互式引导 + 命令行模式。"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

from subtool import __version__
from subtool.core.orchestrator import run_batch
from subtool.hardware.detect import detect_hardware

# 语言别名（与旧版 run.py 兼容）
LANG_ALIASES: dict[str, str] = {
    "zh": "zh", "中文": "zh", "汉语": "zh",
    "en": "en", "英语": "en", "英文": "en",
    "ja": "ja", "日": "ja", "日语": "ja",
    "ko": "ko", "韩": "ko", "韩语": "ko",
    "yue": "yue", "粤": "yue", "粤语": "yue",
    "de": "de", "德": "de", "德语": "de",
    "fr": "fr", "法": "fr", "法语": "fr",
    "es": "es", "西": "es", "西班牙语": "es",
    "it": "it", "意": "it", "意大利语": "it",
    "ru": "ru", "俄": "ru", "俄语": "ru",
}


def normalize_language(raw: str | None) -> str | None:
    if raw is None:
        return None
    s = raw.strip().lower()
    if not s or s == "auto":
        return None
    return LANG_ALIASES.get(s, s)


def format_timedelta(seconds: float) -> str:
    if seconds is None:
        return "未知"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def create_runner(name: str, model_path: str | None = None, model_size: str | None = None):
    """根据名称加载 STT runner。"""
    if name in ("qwen3-asr", "mlx", "qwen", "qwen-asr"):
        from subtool.runners.mlx_qwen3 import MlxQwen3Runner
        return MlxQwen3Runner(
            model_path=Path(model_path) if model_path else None,
            model_size=model_path,  # CLI 中 --model-size 通过 model_path 传入
        )
    raise ValueError(f"未知后端: {name}")


def interactive_prompt() -> dict:
    """交互式收集参数。"""
    print("-" * 40)
    print(f"Subtool v{__version__} — 本地批量转录")
    print("-" * 40)

    # 输入路径
    while True:
        raw = input("\n请输入文件或文件夹路径: ").strip().strip('"\'')
        if not raw:
            continue
        path = Path(raw).expanduser()
        try:
            path = path.resolve()
        except Exception:
            pass
        if path.exists():
            break
        print(f"路径不存在: {path}")

    # 语言
    print("\n语言选项: auto (自动检测), zh (中文), en (英语), ja, ko, yue (粤语), ...")
    raw_lang = input("语言 (默认 auto): ").strip()
    language = normalize_language(raw_lang) if raw_lang else None

    # 输出格式
    fmt = input("输出格式 (srt 或 srt, txt, 默认 srt): ").strip().lower() or "srt"
    formats = [p.strip() for p in fmt.split(",") if p.strip() in {"srt", "txt"}]
    if not formats:
        formats = ["srt"]

    # TXT 时间戳
    timestamp_in_txt = False
    if "txt" in formats:
        ts = input("TXT 是否包含时间戳 (y/N): ").strip().lower()
        timestamp_in_txt = ts in {"y", "yes"}

    # 输出目录
    raw_out = input("输出目录 (留空则与源文件同级): ").strip().strip('"\'')
    output_dir = Path(raw_out).expanduser() if raw_out else None
    if output_dir is not None:
        try:
            output_dir = output_dir.resolve()
        except Exception:
            pass

    # 跳过已有
    skip = input("跳过已有输出的文件 (Y/n): ").strip().lower()
    skip_existing = skip not in {"n", "no"}

    # 后端
    print("\n可用后端: qwen-asr (默认), faster-whisper")
    raw_backend = input("后端 (默认 qwen-asr): ").strip() or "qwen-asr"

    # Dry run
    dry = input("先预览不执行 (y/N): ").strip().lower()
    dry_run = dry in {"y", "yes"}

    # 模型大小
    print("\n模型大小: 0.6B-4bit (最快), 0.6B-8bit, 1.7B-4bit, 1.7B-8bit, 1.7B (最准)")
    raw_size = input("模型 (默认 0.6B-4bit): ").strip() or "0.6B-4bit"

    return {
        "input_path": path,
        "language": language,
        "output_formats": formats,
        "timestamp_in_txt": timestamp_in_txt,
        "output_dir": output_dir,
        "skip_existing": skip_existing,
        "backend": raw_backend,
        "dry_run": dry_run,
        "model_size": raw_size,
    }


def execute_task(config: dict) -> int:
    """执行转录任务。"""
    input_path: Path = config["input_path"]
    output_dir: Path | None = config.get("output_dir")
    output_formats: list[str] = config["output_formats"]
    language: str | None = config.get("language")
    skip_existing: bool = config.get("skip_existing", True)
    timestamp_in_txt: bool = config.get("timestamp_in_txt", False)
    backend_name: str = config.get("backend", "qwen-asr")
    dry_run: bool = config.get("dry_run", False)
    model_size: str | None = config.get("model_size")

    # 检测硬件
    hw = detect_hardware()
    print(f"\n设备: {hw.device}, 内存: {hw.memory_bytes / (1024**3):.1f} GB")

    # 加载 runner
    print(f"加载后端: {backend_name} ...")
    runner = create_runner(backend_name, model_size=model_size)
    print(f"后端就绪: {runner.name}")

    # 扫描文件
    from subtool.output.paths import find_input_files
    all_files = find_input_files(input_path)

    # 计算输出路径，过滤已有
    from subtool.output.paths import compute_output_paths, should_skip_outputs
    planned: list[tuple[Path, dict[str, Path]]] = []
    for f in all_files:
        outputs = compute_output_paths(
            input_path=input_path,
            source_file=f,
            output_dir=output_dir,
            output_formats=output_formats,
        )
        if skip_existing and should_skip_outputs(outputs):
            continue
        planned.append((f, outputs))

    if not planned:
        print("\n所有文件均已处理完成！")
        return 0

    # Dry run
    if dry_run:
        print(f"\n{'─' * 40}")
        print(f"预览: 将处理 {len(planned)} 个文件")
        print(f"{'─' * 40}")
        for idx, (src, outputs) in enumerate(planned, 1):
            print(f"[{idx}/{len(planned)}] {src}")
            for fmt, out in outputs.items():
                print(f"  → {fmt}: {out}")
        return 0

    # 执行
    print(f"\n开始处理 {len(planned)} 个文件...\n")

    stop_event = threading.Event()

    def on_file_done(fr):
        status_icon = {"success": "✓", "skipped": "⊘", "failed": "✗"}
        icon = status_icon.get(fr.status, "?")
        msg = f" ({fr.message})" if fr.message else ""
        print(f"  {icon} {fr.source.name}{msg}")

    result = run_batch(
        input_path=input_path,
        output_dir=output_dir,
        output_formats=output_formats,
        runner=runner,
        language=language,
        skip_existing=skip_existing,
        timestamp_in_txt=timestamp_in_txt,
        on_file_done=on_file_done,
        stop_check=stop_event.is_set,
    )

    print(f"\n{'─' * 40}")
    print("处理完成！")
    print(f"{'─' * 40}")
    print(f"成功: {result.success_count}, 跳过: {result.skip_count}, 失败: {result.fail_count}")

    if result.fail_count > 0:
        print("\n[失败文件]")
        for fr in result.files:
            if fr.status == "failed":
                print(f"  ✗ {fr.source.name}: {fr.message}")

    return 0 if result.fail_count == 0 else 1


def main() -> int:
    """CLI 入口点。"""
    parser = argparse.ArgumentParser(
        prog="subtool",
        description="本地批量音视频转录工具",
    )
    parser.add_argument("--version", action="version", version=f"subtool {__version__}")
    parser.add_argument("input_path", nargs="?", help="输入文件或目录")
    parser.add_argument("--lang", default=None, help="语言代码 (默认 auto)")
    parser.add_argument("--format", default="srt", help="输出格式: srt 或 srt,txt")
    parser.add_argument("--timestamp-in-txt", action="store_true", help="TXT 包含时间戳")
    parser.add_argument("--output-dir", default=None, help="输出目录")
    parser.add_argument("--no-skip-existing", action="store_true", help="不跳过已有输出")
    parser.add_argument("--backend", default="qwen-asr", help="STT 后端 (默认 qwen-asr)")
    parser.add_argument("--model-size", default=None, help="模型大小: 0.6B-4bit, 0.6B-8bit, 0.6B, 1.7B-4bit, 1.7B-8bit, 1.7B")
    parser.add_argument("--dry-run", action="store_true", help="仅预览不执行")
    parser.add_argument("--interactive", "-i", action="store_true", help="交互模式")

    args = parser.parse_args()

    # 决定模式
    if args.interactive or args.input_path is None:
        config = interactive_prompt()
    else:
        output_formats = [p.strip() for p in args.format.split(",") if p.strip() in {"srt", "txt"}]
        if not output_formats:
            output_formats = ["srt"]

        output_dir = Path(args.output_dir) if args.output_dir else None
        if output_dir is not None:
            output_dir = output_dir.expanduser()
            try:
                output_dir = output_dir.resolve()
            except Exception:
                pass

        config = {
            "input_path": Path(args.input_path).expanduser().resolve(),
            "language": normalize_language(args.lang),
            "output_formats": output_formats,
            "timestamp_in_txt": args.timestamp_in_txt,
            "output_dir": output_dir,
            "skip_existing": not args.no_skip_existing,
            "backend": args.backend,
            "dry_run": args.dry_run,
            "model_size": args.model_size,
        }

    return execute_task(config)


if __name__ == "__main__":
    raise SystemExit(main())
