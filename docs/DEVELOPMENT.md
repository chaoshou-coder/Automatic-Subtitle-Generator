# 开发者文档

## 代码结构（核心入口）

- `run.py`：用户入口（依赖校验/安装、环境引导、交互收集参数）
- `batch_transcribe.py`：执行器（无交互，纯批处理逻辑）
- `download_model.py`：可选的模型预下载脚本
- `docs/`：文档

## 发布注意事项

- 不要提交大体积文件/目录：`models/`、`ffmpeg/`、`logs/`、`_temp_wav_cache/`
- 不要提交转录输出：`*.srt`/`*.txt`/`*.part`
- `.whisperrc` 属于本地配置，不应提交到仓库
