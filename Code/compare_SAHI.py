import torch
import time
import os
from thop import profile
from ultralytics import YOLO

def evaluate_models_final():
    # 1. 精简模型列表，先测这几个核心的
    model_paths = [
        r"D:\Capstone\runs\detect\yolov8s\weights\best.pt",
        r"D:\Capstone\runs\detect\sea_full_distill_v1\weights\best.pt",
        r"D:\Capstone\runs\detect\sea_mini_distill_v1\weights\best.pt",
    ]

    data_yaml = r"D:\Capstone\Code\visdrone.yaml"
    output_file = "compare_HighRes_full.txt" # 结果保存文件

    input_size = (1, 3, 640, 640)
    test_iters = 50
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dummy_input = torch.randn(input_size).to(device)

    # 读取已存在记录
    evaluated_models = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("模型:"):
                    evaluated_models.add(line.strip().replace("模型: ", ""))

    with open(output_file, "a", encoding="utf-8") as f:
        for model_path in model_paths:
            if not os.path.exists(model_path):
                print(f"跳过不存在路径: {model_path}")
                continue
            if model_path in evaluated_models:
                print(f"跳过已评估模型: {model_path}")
                continue

            print(f"\n🚀 正在高分辨率分析: {os.path.basename(model_path)}")

            model = YOLO(model_path)
            model.to(device)

            # --- A. 基础参数测试 ---
            total_params = sum(p.numel() for p in model.model.parameters())
            params_m = total_params / 1e6
            flops, _ = profile(model.model, inputs=(dummy_input,), verbose=False)
            flops_g = flops / 1e9

            # --- B. FPS 测试 (保持 640) ---
            model.model.eval()
            if device == "cuda": torch.cuda.synchronize()
            start = time.time()
            with torch.no_grad():
                for _ in range(test_iters): _ = model.model(dummy_input)
            if device == "cuda": torch.cuda.synchronize()
            fps = test_iters / (time.time() - start)

            # --- C. 核心：高分辨率验证 (不使用报错的 tta) ---
            # 1024 分辨率会让小目标像素翻倍，Recall 会大幅提升
            print(f"正在进行 1024 像素深度验证...")
            results = model.val(data=data_yaml, imgsz=1024, verbose=False)

            precision = results.box.p.mean().item()
            recall = results.box.r.mean().item()
            map50 = results.box.map50.mean().item()
            map5095 = results.box.map.mean().item()

            text = (
                f"模型: {model_path}\n"
                f"验证模式: 1024 High-Res (无TTA)\n"
                f"Params: {params_m:.2f} M\n"
                f"FLOPs:  {flops_g:.2f} GFLOPs\n"
                f"FPS:    {fps:.2f}\n"
                f"Precision:  {precision:.4f}\n"
                f"Recall:     {recall:.4f}\n"
                f"mAP50:      {map50:.4f}\n"
                f"mAP50-95:   {map5095:.4f}\n"
                f"{'-'*45}\n"
            )

            print(text)
            f.write(text)

    print(f"\n✨ 完成！结果已保存至 {output_file}")

if __name__ == "__main__":
    evaluate_models_final()