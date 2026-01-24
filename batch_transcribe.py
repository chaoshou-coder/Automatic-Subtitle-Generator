"""
批量转录执行器（无交互）。

关键约定：
- 输出采用 *.part 临时文件并在成功后原子替换为最终文件；
- 默认 skip_existing=True：任一目标输出存在即跳过；
- --output-dir 会保留输入目录相对结构；
- 当前目录的 .whisperrc（JSON）可作为默认参数来源（CLI 优先）。
"""

import argparse
import contextlib
import json
import math
import os
import shutil
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

try:
    import torch
except ImportError:
    torch = None

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".mp4", ".mkv", ".flac", ".mov", ".avi", ".webm"}

VAD_OPTIONS = {
    "min_silence_duration_ms": 1000,
    "speech_pad_ms": 400,
    "min_speech_duration_ms": 250,
    "threshold": 0.5,
}

if sys.platform == "win32":
    os.system("chcp 65001 > nul")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

stop_event = threading.Event()


def signal_handler(sig, frame):
    print("\n[警告] 接收到中断信号，正在完成当前文件后退出...")
    stop_event.set()

LANG_ALIASES: dict[str, str] = {
    "zh": "zh",
    "中文": "zh",
    "汉语": "zh",
    "en": "en",
    "英语": "en",
    "英文": "en",
    "de": "de",
    "德": "de",
    "德语": "de",
    "fr": "fr",
    "法": "fr",
    "法语": "fr",
    "es": "es",
    "西": "es",
    "西语": "es",
    "西班牙": "es",
    "西班牙语": "es",
    "it": "it",
    "意": "it",
    "意语": "it",
    "意大利": "it",
    "意大利语": "it",
    "ru": "ru",
    "俄": "ru",
    "俄语": "ru",
    "俄罗斯": "ru",
    "yue": "yue",
    "粤": "yue",
    "粤语": "yue",
    "ja": "ja",
    "日": "ja",
    "日语": "ja",
    "ko": "ko",
    "韩": "ko",
    "韩语": "ko",
}


def normalize_language(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if not s:
        return None
    if s == "auto":
        return None
    return LANG_ALIASES.get(s, s)


def ensure_ffmpeg():
    if shutil.which("ffmpeg"):
        return True
    doc = Path(__file__).parent / "docs" / "FFMPEG_INSTALL.md"
    print("错误: 未找到 FFmpeg。请先安装 FFmpeg，并确保命令行可用：ffmpeg -version", file=sys.stderr)
    if doc.exists():
        print(f"安装说明: {doc}", file=sys.stderr)
    return False


def format_timestamp_srt(seconds: float) -> str:
    """将秒数格式化为 SRT 时间戳：HH:MM:SS,mmm（毫秒四舍五入，且对异常值兜底）。"""
    try:
        seconds_f = float(seconds)
    except Exception:
        seconds_f = 0.0

    if math.isnan(seconds_f):
        seconds_f = 0.0

    if seconds_f < 0:
        total_ms = int(seconds_f * 1000 - 0.5)
    else:
        total_ms = int(seconds_f * 1000 + 0.5)
    if total_ms < 0:
        total_ms = 0

    h, rem_ms = divmod(total_ms, 3_600_000)
    m, rem_ms = divmod(rem_ms, 60_000)
    s, ms = divmod(rem_ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def format_timedelta_cn(seconds):
    if seconds is None:
        return "未知"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def parse_output_formats(raw: Optional[str]) -> list[str]:
    if not raw:
        return ["srt"]
    valid = {"srt", "txt"}
    items = [p.strip().lower() for p in raw.split(",") if p.strip()]
    deduped: list[str] = []
    for it in items:
        if it in valid and it not in deduped:
            deduped.append(it)
    return deduped or ["srt"]


def should_skip_file(file_path: Path, output_formats: Sequence[str]) -> bool:
    return any(file_path.with_suffix(f".{fmt}").exists() for fmt in output_formats)

def compute_output_paths(
    *,
    input_path: Path,
    source_file: Path,
    output_dir: Optional[Path],
    output_formats: Sequence[str],
) -> dict[str, Path]:
    """
    计算某个源文件对应的输出路径（srt/txt）。

    - 未指定 output_dir：输出与源文件同目录同名
    - 指定 output_dir：输出在 output_dir 内，并尽量保留源文件相对 input_path 的目录结构
    """
    input_root = input_path if input_path.is_dir() else input_path.parent
    try:
        rel = source_file.relative_to(input_root)
    except Exception:
        rel = source_file.name

    base = (output_dir / rel) if output_dir is not None else source_file
    out: dict[str, Path] = {}
    for fmt in output_formats:
        if fmt in {"srt", "txt"}:
            out[fmt] = base.with_suffix(f".{fmt}")
    return out


def should_skip_outputs(output_paths: dict[str, Path]) -> bool:
    return any(p.exists() for p in output_paths.values())


def load_config(config_path: Path) -> dict[str, Any]:
    try:
        raw = config_path.read_text(encoding="utf-8")
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def resolve_config_path(cli_value: Optional[str]) -> Optional[Path]:
    if cli_value:
        return Path(cli_value).expanduser()
    default_path = Path.cwd() / ".whisperrc"
    return default_path if default_path.exists() else None


def find_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in SUPPORTED_EXTENSIONS else []
    files: list[Path] = []
    for p in input_path.rglob("*"):
        try:
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(p)
        except Exception:
            continue
    return files


def detect_device_and_compute_type() -> tuple[str, str]:
    """
    检测优先设备与计算精度。

    规则：
    - 若 CTranslate2 能检测到 CUDA 设备，则使用 cuda/float16；
    - 否则使用 cpu/int8（兼顾速度与兼容性）。
    """
    try:
        import ctranslate2

        if getattr(ctranslate2, "get_cuda_device_count", None) and ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"

def resolve_local_model_dir(models_root: Path) -> Optional[Path]:
    """
    在 models/ 下定位实际模型目录。

    兼容两种布局：
    - 直接目录：models/model.bin
    - snapshot 目录：models/**/model.bin（优先返回同级存在 config.json 的目录）
    """
    direct = models_root / "model.bin"
    if direct.exists():
        return models_root
    try:
        candidates: list[Path] = []
        for p in models_root.rglob("model.bin"):
            parent = p.parent
            if (parent / "config.json").exists():
                return parent
            candidates.append(parent)
        return candidates[0] if candidates else None
    except Exception:
        return None


def _import_whisper_model():
    """延迟导入 faster_whisper.WhisperModel，缺依赖时给出可操作的错误提示。"""
    try:
        from faster_whisper import WhisperModel

        return WhisperModel
    except ImportError:
        print("未找到 faster-whisper。请先运行 run.py 自动安装依赖，或执行：pip install faster-whisper", file=sys.stderr)
        raise


def _create_whisper_model(
    *,
    model_path: str,
    device: str,
    compute_type: str,
    cpu_threads: int,
    download_root: Optional[str],
    gpu_index: Optional[int],
) -> Any:
    """创建 WhisperModel 实例（封装参数拼装，便于统一维护）。"""
    WhisperModel = _import_whisper_model()
    kwargs: dict[str, Any] = {
        "device": device,
        "compute_type": compute_type,
        "cpu_threads": cpu_threads,
        "download_root": download_root,
    }
    if device == "cuda":
        kwargs["device_index"] = int(gpu_index) if gpu_index is not None else 0
    return WhisperModel(model_path, **kwargs)


def load_model(*, gpu_index: Optional[int] = None, cpu_threads: Optional[int] = None) -> Any:
    """
    加载模型：优先使用本地 models/，否则自动下载 large-v3 到 models/ 作为缓存。

    失败回退策略：
    - CPU + int8 失败时，回退到 CPU + float32（更兼容但更慢）。
    """
    device, compute_type = detect_device_and_compute_type()
    local_model_dir = Path(__file__).parent / "models"
    resolved = resolve_local_model_dir(local_model_dir) if local_model_dir.exists() else None
    if resolved is not None:
        model_path = str(resolved)
        download_root = None
    else:
        model_path = "large-v3"
        download_root = str(local_model_dir)
    resolved_cpu_threads = int(cpu_threads) if cpu_threads is not None else (os.cpu_count() or 8)
    try:
        return _create_whisper_model(
            model_path=model_path,
            device=device,
            compute_type=compute_type,
            cpu_threads=resolved_cpu_threads,
            download_root=download_root,
            gpu_index=gpu_index,
        )
    except Exception:
        if device == "cpu" and compute_type != "float32":
            return _create_whisper_model(
                model_path=model_path,
                device=device,
                compute_type="float32",
                cpu_threads=resolved_cpu_threads,
                download_root=download_root,
                gpu_index=gpu_index,
            )
        raise


def transcribe_file(
    model: Any,
    file_path: Path,
    language: Optional[str],
    beam_size: int,
    on_segment: Optional[Callable[[Any, Any], None]] = None,
) -> tuple[float, float]:
    """
    转录单个文件，并在每个 segment 产生时回调 on_segment 进行写出/进度显示。

    这里不做输出文件处理，保持纯转录职责，便于测试与复用。
    """
    started_at = time.time()
    kwargs: dict[str, Any] = {
        "beam_size": beam_size,
        "language": language,
        "vad_filter": True,
        "vad_parameters": VAD_OPTIONS,
    }
    segments_gen, info = model.transcribe(str(file_path), **kwargs)
    for segment in segments_gen:
        if on_segment is not None:
            on_segment(segment, info)
    return float(getattr(info, "duration", 0.0) or 0.0), time.time() - started_at

def finalize_output_files(temp_to_final: dict[Path, Path]) -> None:
    """将临时输出文件原子替换为最终输出文件。"""
    for tmp, final in temp_to_final.items():
        os.replace(tmp, final)


def cleanup_temp_files(temp_paths: Iterable[Path]) -> None:
    """清理临时文件（尽量不抛异常，避免遮蔽主错误）。"""
    for p in temp_paths:
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass


@dataclass
class ProgressTracker:
    """按 segment 更新并打印“全局 ETA”的轻量进度状态机。"""
    idx: int
    total: int
    file_name: str
    file_started: float
    last_ui_update: float = 0.0
    current_audio_time: float = 0.0
    current_duration: float = 0.0

    def update(self, *, seg_end: float, duration: float, completed_files: int, completed_wall_seconds: float) -> None:
        self.current_audio_time = float(seg_end or 0.0)
        self.current_duration = float(duration or 0.0)
        now = time.time()
        if now - self.last_ui_update < 1.0:
            return
        self.last_ui_update = now

        elapsed_current = now - self.file_started
        remaining_current = 0.0
        if self.current_duration > 0 and self.current_audio_time > 0:
            progress = min(0.999, self.current_audio_time / self.current_duration)
            est_total = elapsed_current / progress
            remaining_current = max(0.0, est_total - elapsed_current)

        remaining_after = max(0, self.total - (completed_files + 1))
        avg_wall = (completed_wall_seconds / completed_files) if completed_files > 0 else 0.0
        eta_total = remaining_current + remaining_after * avg_wall
        print(
            f"\r[{self.idx}/{self.total}] {self.file_name} - 预计剩余 {format_timedelta_cn(eta_total)}",
            end="",
            flush=True,
        )


class SegmentWriter:
    """
    将转录 segment 写出到 SRT/TXT。

    设计为可调用对象，以便直接作为 transcribe_file(on_segment=...) 传入。
    """
    def __init__(
        self,
        *,
        srt_f,
        txt_f,
        timestamp_in_txt: bool,
        tracker: ProgressTracker,
        get_completed_files: Callable[[], int],
        get_completed_wall_seconds: Callable[[], float],
    ) -> None:
        self._srt_f = srt_f
        self._txt_f = txt_f
        self._timestamp_in_txt = bool(timestamp_in_txt)
        self._tracker = tracker
        self._get_completed_files = get_completed_files
        self._get_completed_wall_seconds = get_completed_wall_seconds
        self._srt_index = 0

    def __call__(self, seg: Any, info: Any) -> None:
        """写出单个 segment，并触发进度刷新。"""
        duration = float(getattr(info, "duration", 0.0) or 0.0)
        seg_end = float(getattr(seg, "end", 0.0) or 0.0)
        self._tracker.update(
            seg_end=seg_end,
            duration=duration,
            completed_files=self._get_completed_files(),
            completed_wall_seconds=self._get_completed_wall_seconds(),
        )

        if self._srt_f is not None:
            self._srt_index += 1
            self._srt_f.write(f"{self._srt_index}\n")
            self._srt_f.write(f"{format_timestamp_srt(seg.start)} --> {format_timestamp_srt(seg.end)}\n")
            self._srt_f.write(f"{str(seg.text).strip()}\n\n")

        if self._txt_f is not None:
            if self._timestamp_in_txt:
                self._txt_f.write(f"[{seg.start:.2f} -> {seg.end:.2f}] {seg.text}\n")
            else:
                self._txt_f.write(f"{seg.text}\n")


def _maybe_install_signal_handlers() -> None:
    """尝试注册 SIGINT 处理器（Windows/特定宿主下可能失败，失败时忽略）。"""
    try:
        signal.signal(signal.SIGINT, signal_handler)
    except Exception:
        pass


def _maybe_empty_cuda_cache() -> None:
    """失败重试前清理 CUDA 缓存，避免显存碎片导致连续失败（尽量不抛异常）。"""
    if torch is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _prepare_output_files(
    *,
    output_paths: dict[str, Path],
    output_formats: Sequence[str],
) -> tuple[dict[Path, Path], contextlib.ExitStack, Any, Any]:
    """
    打开临时输出文件并返回：
    - temp_to_final：临时路径 -> 最终路径
    - ExitStack：用于确保异常情况下也能关闭文件句柄
    - srt_f/txt_f：打开后的文件对象（可能为 None）
    """
    temp_to_final: dict[Path, Path] = {}
    stack = contextlib.ExitStack()
    srt_f = None
    txt_f = None

    if "srt" in output_formats:
        final_srt = output_paths["srt"]
        tmp_srt = final_srt.with_suffix(final_srt.suffix + ".part")
        temp_to_final[tmp_srt] = final_srt
        cleanup_temp_files([tmp_srt])
        final_srt.parent.mkdir(parents=True, exist_ok=True)
        srt_f = stack.enter_context(open(tmp_srt, "w", encoding="utf-8"))

    if "txt" in output_formats:
        final_txt = output_paths["txt"]
        tmp_txt = final_txt.with_suffix(final_txt.suffix + ".part")
        temp_to_final[tmp_txt] = final_txt
        cleanup_temp_files([tmp_txt])
        final_txt.parent.mkdir(parents=True, exist_ok=True)
        txt_f = stack.enter_context(open(tmp_txt, "w", encoding="utf-8"))

    return temp_to_final, stack, srt_f, txt_f


def process_single_file(
    *,
    model: Any,
    idx: int,
    total: int,
    file_path: Path,
    output_paths: dict[str, Path],
    language: Optional[str],
    output_formats: Sequence[str],
    timestamp_in_txt: bool,
    beam_size: int,
    total_attempts: int,
    get_completed_files: Callable[[], int],
    get_completed_wall_seconds: Callable[[], float],
) -> tuple[bool, float]:
    """
    处理单个文件（包含重试与输出文件的清理/落盘）。

    返回：
    - ok：是否成功
    - wall_seconds：本文件处理耗时（用于全局 ETA）
    """
    file_started = time.time()
    tracker = ProgressTracker(idx=idx, total=total, file_name=file_path.name, file_started=file_started)
    last_error: Optional[BaseException] = None

    for attempt in range(total_attempts):
        temp_to_final: dict[Path, Path] = {}
        stack = None
        try:
            temp_to_final, stack, srt_f, txt_f = _prepare_output_files(
                output_paths=output_paths,
                output_formats=output_formats,
            )
            writer = SegmentWriter(
                srt_f=srt_f,
                txt_f=txt_f,
                timestamp_in_txt=timestamp_in_txt,
                tracker=tracker,
                get_completed_files=get_completed_files,
                get_completed_wall_seconds=get_completed_wall_seconds,
            )
            duration, elapsed = transcribe_file(
                model,
                file_path,
                language,
                beam_size=beam_size,
                on_segment=writer,
            )
            stack.close()
            finalize_output_files(temp_to_final)
            print(f"\r[{idx}/{total}] {file_path.name} - 完成 (耗时 {elapsed:.1f}s)                    ")
            return True, time.time() - file_started
        except KeyboardInterrupt as e:
            last_error = e
            try:
                if stack is not None:
                    stack.close()
            except Exception:
                pass
            cleanup_temp_files(temp_to_final.keys())
            raise
        except Exception as e:
            last_error = e
            try:
                if stack is not None:
                    stack.close()
            except Exception:
                pass
            cleanup_temp_files(temp_to_final.keys())
            _maybe_empty_cuda_cache()
            if attempt < total_attempts - 1:
                print(
                    f"\n[警告] 失败，将重试 {attempt+1}/{total_attempts-1}: {file_path.name} - {e}",
                    file=sys.stderr,
                )
                time.sleep(1)
                continue
            print(f"\n[错误] 失败: {file_path.name} - {e}", file=sys.stderr)
            return False, time.time() - file_started

    if last_error is not None:
        print(f"\n[错误] 失败: {file_path.name} - {last_error}", file=sys.stderr)
    return False, time.time() - file_started


def run_task(
    *,
    input_path: Path,
    language: Optional[str],
    output_formats: list[str],
    timestamp_in_txt: bool,
    fast: bool,
    retries: int,
    beam_size: Optional[int] = None,
    cpu_threads: Optional[int] = None,
    gpu: Optional[int] = None,
    output_dir: Optional[Path] = None,
    skip_existing: bool = True,
    dry_run: bool = False,
) -> None:
    """
    执行批处理任务（主入口）。

    该函数对外作为稳定 API 被 run.py 调用，也被 CLI main() 调用。
    """
    stop_event.clear()
    if not ensure_ffmpeg():
        raise SystemExit(2)

    if not input_path.exists():
        print(f"路径不存在: {input_path}", file=sys.stderr)
        raise SystemExit(2)

    final_output_formats = [f for f in output_formats if f in {"srt", "txt"}] or ["srt"]

    files_to_process = find_input_files(input_path)
    output_dir_resolved = None
    if output_dir is not None:
        output_dir_resolved = Path(output_dir).expanduser()
        try:
            output_dir_resolved = output_dir_resolved.resolve()
        except Exception:
            pass

    planned: list[tuple[Path, dict[str, Path]]] = []
    for p in files_to_process:
        outputs = compute_output_paths(
            input_path=input_path,
            source_file=p,
            output_dir=output_dir_resolved,
            output_formats=final_output_formats,
        )
        if skip_existing and should_skip_outputs(outputs):
            continue
        planned.append((p, outputs))

    final_files = [p for p, _ in planned]
    if not final_files:
        print("所有文件均已处理完成！")
        return

    if dry_run:
        print("-" * 30)
        print(f"Dry Run: 将处理 {len(planned)} 个文件")
        print("-" * 30)
        for idx, (p, outputs) in enumerate(planned, 1):
            print(f"[{idx}/{len(planned)}] {p}")
            for fmt, out in outputs.items():
                print(f"  -> {fmt}: {out}")
        return

    print(f"任务开始: {len(final_files)} 个文件待处理")

    success_count = 0
    fail_count = 0
    failed_files: list[str] = []

    resolved_beam_size = int(beam_size) if beam_size is not None else (1 if fast else 5)
    total_attempts = 1 + max(0, int(retries))

    model = load_model(gpu_index=gpu, cpu_threads=cpu_threads)

    _maybe_install_signal_handlers()
    global_start = time.time()
    completed_files = 0
    completed_wall_seconds = 0.0
    def get_completed_files() -> int:
        return completed_files

    def get_completed_wall_seconds() -> float:
        return completed_wall_seconds

    try:
        for idx, (file_path, output_paths) in enumerate(planned, 1):
            if stop_event.is_set():
                break

            print(f"\n[{idx}/{len(final_files)}] {file_path.name} - 开始")
            ok, wall_elapsed = process_single_file(
                model=model,
                idx=idx,
                total=len(final_files),
                file_path=file_path,
                output_paths=output_paths,
                language=language,
                output_formats=final_output_formats,
                timestamp_in_txt=timestamp_in_txt,
                beam_size=resolved_beam_size,
                total_attempts=total_attempts,
                get_completed_files=get_completed_files,
                get_completed_wall_seconds=get_completed_wall_seconds,
            )
            completed_files += 1
            completed_wall_seconds += wall_elapsed
            if ok:
                success_count += 1
            else:
                fail_count += 1
                failed_files.append(file_path.name)
    except KeyboardInterrupt:
        print("\n[警告] 接收到中断信号，将结束当前任务。", file=sys.stderr)
        stop_event.set()

    elapsed_all = time.time() - global_start
    print("\n" + "-" * 30)
    print("处理完成！")
    print("-" * 30)
    print(f"成功: {success_count}, 失败: {fail_count}")
    print(f"总耗时: {format_timedelta_cn(elapsed_all)}")
    if failed_files:
        print("\n[失败文件列表]")
        for f in failed_files:
            print(f"- {f}")

# ============================================
# 主程序
# ============================================
def parse_args():
    """解析 CLI 参数（CLI 仅覆盖必要值，其他可从 .whisperrc 获取默认值）。"""
    parser = argparse.ArgumentParser(description="批量转录音视频文件")
    parser.add_argument("input_path", help="目录或文件路径")
    parser.add_argument("--lang", default=None, help="语言代码: zh/en/auto 等 (默认 zh)")
    parser.add_argument("--format", default=None, help="输出格式: srt 或 srt,txt (默认 srt)")
    parser.add_argument("--timestamp-in-txt", dest="timestamp_in_txt", action="store_true", default=None, help="TXT 输出包含时间戳")
    parser.add_argument("--fast", action="store_true", default=None, help="快速模式：beam_size=1")
    parser.add_argument("--beam-size", type=int, default=None, help="beam_size (默认 5；--fast 时默认 1)")
    parser.add_argument("--cpu-threads", type=int, default=None, help="CPU 线程数 (默认 os.cpu_count())")
    parser.add_argument("--gpu", type=int, default=None, help="指定 GPU 序号 (仅在 CUDA 可用时生效)")
    parser.add_argument("--retries", type=int, default=None, help="失败重试次数(默认0)")
    parser.add_argument("--output-dir", default=None, help="输出目录（默认写入源文件同级目录）")
    parser.add_argument("--dry-run", action="store_true", default=None, help="仅预览将处理的文件与输出路径，不执行转录")
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=None, help="跳过已有输出的文件（默认开启）")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false", default=None, help="不跳过已有输出（覆盖重新生成）")
    parser.add_argument("--config", default=None, help="配置文件路径（默认自动读取当前目录的 .whisperrc）")
    return parser.parse_args()

def main():
    """CLI 入口：合并配置文件与命令行参数，然后调用 run_task。"""
    args = parse_args()
    config_path = resolve_config_path(args.config)
    config = load_config(config_path) if config_path is not None else {}

    input_path = Path(args.input_path)
    try:
        input_path = input_path.expanduser().resolve()
    except Exception:
        pass

    raw_lang = args.lang if args.lang is not None else config.get("lang", "zh")
    language = normalize_language(raw_lang)

    if args.format is not None:
        output_formats = parse_output_formats(args.format)
    elif "output_formats" in config:
        output_formats = [str(x).strip().lower() for x in (config.get("output_formats") or []) if str(x).strip()]
        output_formats = [f for f in output_formats if f in {"srt", "txt"}] or ["srt"]
    elif "format" in config:
        output_formats = parse_output_formats(str(config.get("format")))
    else:
        output_formats = ["srt"]

    output_dir = args.output_dir if args.output_dir is not None else config.get("output_dir")
    dry_run = bool(args.dry_run) if args.dry_run is not None else bool(config.get("dry_run", False))
    skip_existing = bool(args.skip_existing) if args.skip_existing is not None else bool(config.get("skip_existing", True))
    fast = bool(args.fast) if args.fast is not None else bool(config.get("fast", False))
    timestamp_in_txt = bool(args.timestamp_in_txt) if args.timestamp_in_txt is not None else bool(config.get("timestamp_in_txt", False))
    retries = int(args.retries) if args.retries is not None else int(config.get("retries", 0) or 0)
    beam_size = args.beam_size if args.beam_size is not None else config.get("beam_size")
    cpu_threads = args.cpu_threads if args.cpu_threads is not None else config.get("cpu_threads")
    gpu = args.gpu if args.gpu is not None else config.get("gpu")

    run_task(
        input_path=input_path,
        language=language,
        output_formats=output_formats,
        timestamp_in_txt=timestamp_in_txt,
        fast=fast,
        retries=retries,
        beam_size=beam_size,
        cpu_threads=cpu_threads,
        gpu=gpu,
        output_dir=Path(output_dir) if output_dir else None,
        skip_existing=skip_existing,
        dry_run=dry_run,
    )

if __name__ == "__main__":
    main()
