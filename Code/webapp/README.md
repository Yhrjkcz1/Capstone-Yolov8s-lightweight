# YOLOv8s-GEA · VisDrone Gradio Demo

面向毕业设计的本地 Web 演示：在 **GitHub 文档风**界面中加载 **YOLOv8s + GEA + 知识蒸馏** 权重，对无人机航拍场景（VisDrone）图像做交互式检测与结果表格展示。

**截图占位**：运行 `app.py` 后自行截图并插入此处。

## Quick Start

### 1. 安装依赖

建议使用虚拟环境（Python 3.10+）：

```bash
cd D:\Capstone\Code\webapp
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 放置权重

确认权重路径与 `app.py` 中 `WEIGHTS_PATH` 一致（默认）：

`D:\Capstone\runs\detect\yolov8s_GEA(KD)\weights\best.pt`

### 3. 启动

```bash
python app.py
```

浏览器打开：`http://127.0.0.1:7860`

可选参数：

- `--port`：端口（默认 `7860`）
- `--share`：生成 Gradio 公网分享链接

### 4. 示例图

将示例图片放入 `assets/samples/`，页面底部 **Examples** 会自动加载常见后缀（jpg/png/webp 等）。

## 文件结构

```text
webapp/
├── app.py                  # Gradio 主程序（全局加载模型、UI、FPS 实测）
├── inference.py            # Detector + safe_imread/safe_imwrite
├── requirements.txt
├── README.md
├── assets/
│   ├── custom.css          # GitHub 文档风样式
│   └── samples/            # 预置示例图（自备）
└── .gitignore
```

## 部署到 Hugging Face Spaces（简要）

1. 在 Hugging Face 新建 **Space**，模板选 **Gradio**，准备好 `requirements.txt` 与 `app.py`。
2. 将本仓库（或 `webapp` 目录）推送到 Space 的 Git 仓库；确保 **`app.py` 入口**与 Space 设置一致（一般为根目录 `app.py`）。
3. 上传或绑定权重：小模型可放入仓库；大权重推荐 **Git LFS**，或使用 **Space Secrets** 在启动脚本中下载到本地路径。
4. 修改 `WEIGHTS_PATH`（或通过环境变量读取）指向 Space 内权重文件；必要时将 `device` 设为 `cpu`。
5. 在 Space **README** 中写清许可证与引用；首次构建完成后检查 **Logs** 排查依赖或 CUDA 相关问题。

---

联系方式与引用格式见页面底部 **About** 区块（可自行修改 `app.py` 中 Markdown）。
