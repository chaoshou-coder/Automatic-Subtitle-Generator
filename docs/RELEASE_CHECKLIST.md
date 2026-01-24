# GitHub 发布检查清单

## 1. 仓库内容检查

- 不要提交大体积文件/目录：`models/`、`ffmpeg/`、`logs/`、`_temp_wav_cache/`
- 确保示例命令能在干净环境运行

## 2. 文档一致性

- README 与 PROJECT_MANUAL 的参数、默认值一致
- README 的“快速开始”入口为 `python run.py`
- 文档中已明确 FFmpeg 安装方式，并提供 `docs/FFMPEG_INSTALL.md`
- PROJECT_MANUAL 已说明：
  - 断点续传规则（任一目标输出存在即跳过）
  - `.part` 临时文件行为

## 3. 测试与报告

建议做一次最小验证：
- `python -m py_compile run.py batch_transcribe.py download_model.py`
- `python batch_transcribe.py <input_path> --dry-run --output-dir out_subtitles`

## 4. 版本与变更记录

- 在 `CHANGELOG.md` 的 `Unreleased` 里补齐这次发布的新增/变更/修复
- 将 `Unreleased` 内容移动到新版本号（例如 `0.1.1`），并清空 `Unreleased`

## 5. Release 文案（建议）

- 亮点：默认 srt、断点续传不覆盖、--fast 提速、多卡选择
