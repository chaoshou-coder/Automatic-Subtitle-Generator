# FFmpeg 安装指南

本项目需要 FFmpeg 用于解码音视频。请确保以下命令可用：

```bash
ffmpeg -version
```

## Windows（推荐）

### 方式 1：手动安装（通用）
1) 访问 FFmpeg 官方下载页（或你信任的发行版下载页）下载 Windows build  
2) 解压到例如：`C:\ffmpeg\`  
3) 将 `C:\ffmpeg\bin\` 加入系统环境变量 PATH  
4) 重新打开终端，执行 `ffmpeg -version` 验证

### 方式 2：使用 winget（如可用）
```powershell
winget install -e --id Gyan.FFmpeg
```

## macOS

### Homebrew（推荐）
```bash
brew install ffmpeg
```

## Linux

### Debian/Ubuntu
```bash
sudo apt update
sudo apt install -y ffmpeg
```

### CentOS/RHEL/Fedora
不同发行版仓库策略不一，请优先使用系统包管理器安装 `ffmpeg`，或按发行版文档启用对应源。

