"""
交互入口（推荐给普通用户）。

职责：
- 依赖/FFmpeg 检测与引导；
- 交互收集参数（含 dry-run 预览）；
- 调用 batch_transcribe.run_task 执行批处理；
- 可读写当前目录 .whisperrc（JSON）保存常用默认参数。
"""

import os
import shutil
import subprocess
import sys
import json
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent


def run_subprocess(args: list[str]) -> int:
    return subprocess.call(args, cwd=str(PROJECT_ROOT))


def run_subprocess_checked(args: list[str]) -> None:
    completed = subprocess.run(args, cwd=str(PROJECT_ROOT))
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def prompt_text(question: str, *, default: Optional[str] = None) -> str:
    suffix = f" (默认 {default})" if default is not None else ""
    while True:
        value = input(f"{question}{suffix}: ").strip()
        if value:
            return value
        if default is not None:
            return default


def prompt_yes_no(question: str, *, default_yes: bool) -> bool:
    hint = "Y/n" if default_yes else "y/N"
    value = input(f"{question} ({hint}): ").strip().lower()
    if not value:
        return default_yes
    return value in {"y", "yes"}


def ensure_pip_available() -> None:
    try:
        import ensurepip

        ensurepip.bootstrap(upgrade=True)
    except Exception as e:
        print(f"警告：ensurepip 运行失败：{e}", file=sys.stderr)

    completed = subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True, text=True)
    if completed.returncode != 0:
        print("未检测到 pip。请先安装 Python（包含 pip），或执行：python -m ensurepip --upgrade", file=sys.stderr)
        raise SystemExit(2)


def pip_install(packages: list[str]) -> None:
    run_subprocess_checked([sys.executable, "-m", "pip", "install", *packages])


def ensure_python_deps() -> None:
    try:
        import faster_whisper  # noqa: F401

        return
    except Exception:
        pass

    print("检测到未安装 faster-whisper，正在自动安装（可能需要几分钟）...")
    ensure_pip_available()
    pip_install(["-U", "pip"])
    pip_install(["faster-whisper"])


def has_local_model(models_dir: Path) -> bool:
    """判断 models/ 下是否已有可用模型（粗略检查 model.bin）。"""
    if not models_dir.exists():
        return False
    direct = models_dir / "model.bin"
    if direct.exists():
        return True
    try:
        return any(models_dir.rglob("model.bin"))
    except Exception:
        return False


def maybe_pre_download_model() -> None:
    """若未检测到本地模型，则询问用户是否预下载 large-v3 到 models/ 作为缓存。"""
    models_dir = PROJECT_ROOT / "models"
    if has_local_model(models_dir):
        print(f"已检测到本地模型目录：{models_dir}")
        return

    wants = prompt_yes_no("是否预先下载模型到 models/（约 3GB）", default_yes=False)
    if not wants:
        return

    use_hf_transfer = prompt_yes_no("是否安装 hf_transfer 以加速下载（可选）", default_yes=False)
    if use_hf_transfer:
        ensure_pip_available()
        pip_install(["hf_transfer"])

    print("开始下载模型...")
    run_subprocess_checked([sys.executable, str(PROJECT_ROOT / "download_model.py")])


def prompt_int(question: str, *, default: Optional[int] = None, allow_blank: bool = False) -> Optional[int]:
    """读取整数输入并做校验，避免 ValueError 导致交互流程崩溃。"""
    while True:
        suffix = ""
        if allow_blank and default is None:
            suffix = "（留空跳过）"
        elif default is not None:
            suffix = f"（默认 {default}）"

        raw = input(f"{question}{suffix}: ").strip()
        if not raw:
            if allow_blank:
                return default
            if default is not None:
                return default
            print("请输入数字。", file=sys.stderr)
            continue
        try:
            return int(raw)
        except Exception:
            print("请输入合法整数。", file=sys.stderr)


def collect_task_config() -> dict:
    """
    收集并返回执行器所需的任务配置 dict。

    返回的字段会直接透传给 batch_transcribe.run_task()。
    """
    import batch_transcribe as bt

    config_path = bt.resolve_config_path(None)
    config = bt.load_config(config_path) if config_path is not None else {}

    raw_input_path = prompt_text("请输入文件或文件夹路径").strip().strip('"\'')
    input_path = Path(raw_input_path).expanduser()
    try:
        input_path = input_path.resolve()
    except Exception:
        pass
    if not input_path.exists():
        print(f"路径不存在: {input_path}", file=sys.stderr)
        raise SystemExit(2)

    print("请选择语言（可输入 auto），常用选项：")
    print("- auto：自动检测")
    print("- zh：中文（简体/繁体）")
    print("- en：English")
    print("- de：Deutsch（德语）")
    print("- fr：Français（法语）")
    print("- es：Español（西班牙语）")
    print("- it：Italiano（意大利语）")
    print("- ru：Русский（俄语）")
    print("- yue：粤语")
    print("- ja：日本語")
    print("- ko：한국어")
    default_lang = str(config.get("lang", "zh"))
    raw_lang = prompt_text("语言（代码或中文名称）", default=default_lang).strip()
    lang = bt.normalize_language(raw_lang) or bt.normalize_language("zh")

    fmt = prompt_text("输出格式（srt 或 srt,txt）", default="srt").strip().lower()
    formats = [p.strip() for p in fmt.split(",") if p.strip()]
    formats = [f for f in formats if f in {"srt", "txt"}]
    if not formats:
        formats = ["srt"]

    if formats == ["txt"]:
        print("提示：TXT 是纯文本转写，不是字幕格式，无法直接用于 FFmpeg 烧录。")
        if prompt_yes_no("是否同时输出 SRT 字幕文件（推荐）", default_yes=True):
            formats = ["srt", "txt"]

    timestamp_in_txt = False
    if "txt" in formats:
        timestamp_in_txt = prompt_yes_no("TXT 是否包含时间戳", default_yes=bool(config.get("timestamp_in_txt", False)))

    fast = prompt_yes_no("启用快速模式（beam_size=1）", default_yes=bool(config.get("fast", True)))

    output_dir_raw = prompt_text("输出目录（留空写入源文件同级）", default=str(config.get("output_dir") or "")).strip().strip('"\'')
    output_dir = Path(output_dir_raw).expanduser() if output_dir_raw else None
    if output_dir is not None:
        try:
            output_dir = output_dir.resolve()
        except Exception:
            pass

    skip_existing = prompt_yes_no("跳过已有输出的文件（推荐）", default_yes=bool(config.get("skip_existing", True)))
    dry_run = prompt_yes_no("是否先预览将处理的文件（dry-run）", default_yes=False)

    advanced = prompt_yes_no("是否打开高级设置", default_yes=False)
    retries = 0
    beam_size = None
    cpu_threads = None
    gpu = None
    if advanced:
        retries = int(prompt_int("失败重试次数", default=int(config.get("retries", 0) or 0)) or 0)
        beam_size = prompt_int("beam_size", default=None, allow_blank=True)
        cpu_threads = prompt_int("cpu_threads", default=None, allow_blank=True)
        gpu = prompt_int("gpu 序号", default=None, allow_blank=True)

    return {
        "input_path": input_path,
        "language": lang,
        "output_formats": formats,
        "timestamp_in_txt": timestamp_in_txt,
        "fast": fast,
        "retries": retries,
        "beam_size": beam_size,
        "cpu_threads": cpu_threads,
        "gpu": gpu,
        "output_dir": output_dir,
        "skip_existing": skip_existing,
        "dry_run": dry_run,
        "_config": config,
        "_config_path": config_path,
    }


def main() -> int:
    """程序入口：环境校验 -> 可选预下载 -> 交互收集参数 -> 调用执行器。"""
    if sys.version_info < (3, 8):
        print("需要 Python 3.8+。请先安装/升级 Python 后再运行。", file=sys.stderr)
        return 2

    print("-" * 30)
    print("Faster Whisper 批量转录工具")
    print("-" * 30)

    ensure_python_deps()
    import batch_transcribe as bt

    if not bt.ensure_ffmpeg():
        return 2
    maybe_pre_download_model()

    config = collect_task_config()

    config_path = config.pop("_config_path", None)
    existing_config = config.pop("_config", {})

    if config.get("dry_run"):
        bt.run_task(**config)
        if not prompt_yes_no("是否开始执行转录", default_yes=False):
            return 0
        config["dry_run"] = False

    bt.run_task(**config)

    to_save = {k: v for k, v in config.items() if k not in {"input_path", "dry_run"}}
    if to_save.get("output_dir") is not None:
        to_save["output_dir"] = str(to_save["output_dir"])
    save_default_yes = not bool(existing_config)
    if prompt_yes_no("是否保存配置到 .whisperrc 供下次默认使用", default_yes=save_default_yes):
        target = (Path.cwd() / ".whisperrc") if config_path is None else (Path.cwd() / ".whisperrc")
        target.write_text(json.dumps(to_save, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已保存配置：{target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
