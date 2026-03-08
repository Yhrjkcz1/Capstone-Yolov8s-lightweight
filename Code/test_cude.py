import torch

# 打印 PyTorch 用的 CUDA 版本
print("PyTorch CUDA version:", torch.version.cuda)

# 检查 GPU 是否可用
print("CUDA available:", torch.cuda.is_available())

# 打印当前可用 GPU 数量
print("Number of GPUs:", torch.cuda.device_count())

# 打印第一块 GPU 的名称
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
