## 电脑环境配置输出
import platform
import sys

try:
    import torch
except ImportError:
    torch = None

with open("environment.txt", "w") as f:
    # Python 和系统
    f.write(f"Python: {sys.version.splitlines()[0]}\n")
    f.write(f"OS: {platform.platform()}\n")

    # PyTorch 信息
    if torch is not None:
        f.write(f"PyTorch: {torch.__version__}\n")
        f.write(f"PyTorch CUDA version: {torch.version.cuda}\n")
        f.write(f"CUDA available: {torch.cuda.is_available()}\n")

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            f.write(f"Number of GPUs: {gpu_count}\n")
            for i in range(gpu_count):
                f.write(f"GPU {i}: {torch.cuda.get_device_name(i)}\n")
    else:
        f.write("PyTorch: not installed\n")

print("已输出到 environment.txt")
