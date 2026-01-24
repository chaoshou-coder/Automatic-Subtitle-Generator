"""
模型预下载脚本（可选）：下载 large-v3 到 models/ 作为本地缓存。
"""

import os
import sys

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
# 国内环境可尝试使用镜像：
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

MODEL_DIR = "models"
MODEL_SIZE = "large-v3"

def main():
    try:
        from faster_whisper import download_model
    except ImportError:
        print("未找到 faster-whisper。请先运行 run.py 自动安装依赖，或执行：pip install faster-whisper", file=sys.stderr)
        return 2

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    print(f"Downloading {MODEL_SIZE} to {os.path.abspath(MODEL_DIR)}...")
    try:
        import hf_transfer

        print("已启用 hf_transfer 加速下载。")
    except Exception:
        print("hf_transfer 未安装：将使用普通下载。需要加速可执行：pip install hf_transfer")
    print("下载体积约 3GB，耗时取决于网络环境。")
    
    try:
        model_path = download_model(MODEL_SIZE, output_dir=MODEL_DIR)
        
        print(f"\n[SUCCESS] Model downloaded successfully to: {model_path}")
        print("batch_transcribe.py 会自动在 models/ 下查找 model.bin 并加载。")
        
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        print("\n可能的解决方式：")
        print("1) 检查网络连接/代理设置")
        print("2) 国内环境可尝试启用 HF_ENDPOINT 镜像")
        print("3) 重新执行脚本重试")
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
