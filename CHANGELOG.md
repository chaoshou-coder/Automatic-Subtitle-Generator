# Changelog

## Unreleased

## 0.1.0

### Changed
- 新增 `run.py` 作为用户入口：依赖校验/自动安装、环境引导、交互收集参数后调用执行器
- `batch_transcribe.py` 改为纯执行器（不再包含交互模式）
- 默认输出改为 srt（可选 txt），移除 vtt/json
- 断点续传改为“任一目标输出存在即跳过”
- 默认重试次数改为 0（仍会执行首次尝试）
- 自动检测 CUDA/CPU，无需手动指定 device/compute-type
- 移除日志文件输出与清屏行为，控制台信息更精简
- 新增参数：--fast/--beam-size/--cpu-threads/--gpu
- 改为边转录边写出（.part 临时文件），避免长音频 segments 堆内存
- 支持识别 models/ 下的 snapshot 下载目录结构（与 download_model.py 兼容）
