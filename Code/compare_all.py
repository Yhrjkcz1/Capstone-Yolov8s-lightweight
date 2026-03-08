import torch
import time
import os
from thop import profile
from ultralytics import YOLO

def evaluate_models():
    model_paths = [
        r"D:\Capstone\runs\detect\yolov8n\weights\best.pt",
        r"D:\Capstone\runs\detect\yolov8s\weights\best.pt",
        r"D:\Capstone\runs\detect\yolov8s_ghost\weights\best.pt",
        r"D:\Capstone\runs\detect\yolov8s_ghost_ca\weights\best.pt",
        r"D:\Capstone\runs\detect\yolov8s_sg_2eca\weights\best.pt",
        r"D:\Capstone\runs\detect\yolov8s_mini_fusionCa\weights\best.pt",
        r"D:\Capstone\runs\detect\yolov8s_mini_fusionECA\weights\best.pt",
        r"D:\Capstone\runs\detect\yolov8s_mini_fusionSGECA\weights\best.pt",
        r"D:\Capstone\runs\detect\v8s_afpn_precision\weights\best.pt",
        r"D:\Capstone\runs\detect\v8s_afpn_full_100e_fair\weights\best.pt",
        r"D:\Capstone\runs\detect\sea_full_distill_v1\weights\best.pt",
    ]

    data_yaml = r"D:\Capstone\Code\visdrone.yaml"
    output_file = "compare.txt"

    # --- 新增：读取已评估过的模型列表 ---
    evaluated_models = []
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            content = f.read()
            for path in model_paths:
                if path in content:
                    evaluated_models.append(path)
    
    # 过滤掉已经存在的模型
    models_to_run = [p for p in model_paths if p not in evaluated_models]

    if not models_to_run:
        print("所有模型已评估完毕，没有发现新模型。")
        return

    print(f"发现 {len(models_to_run)} 个新模型需要评估...")

    input_size = (1, 3, 640, 640)
    test_iters = 50
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dummy_input = torch.randn(input_size).to(device)

    # 使用 "a" (append) 模式打开文件，避免覆盖之前的数据
    with open(output_file, "a", encoding="utf-8") as f:
        for model_path in models_to_run:
            if not os.path.exists(model_path):
                print(f"跳过不存在的文件: {model_path}")
                continue

            print(f"\n正在分析新模型: {model_path}")
            model = YOLO(model_path)
            model.to(device)

            # 计算参数量
            total_params = sum(p.numel() for p in model.model.parameters())
            params_m = total_params / 1e6

            # 计算 FLOPs
            flops, _ = profile(model.model, inputs=(dummy_input,), verbose=False)
            flops_g = flops / 1e9

            # 测试 FPS
            model.model.eval()
            if device == "cuda":
                torch.cuda.synchronize()
            
            start = time.time()
            with torch.no_grad(): # 增加推理加速
                for _ in range(test_iters):
                    _ = model.model(dummy_input)
            
            if device == "cuda":
                torch.cuda.synchronize()
            fps = test_iters / (time.time() - start)

            # 验证模型精度
            results = model.val(data=data_yaml, imgsz=640, verbose=False)
            precision = results.box.p.mean().item()
            recall = results.box.r.mean().item()
            map50 = results.box.map50.mean().item()
            map5095 = results.box.map.mean().item()

            text = (
                f"模型: {model_path}\n"
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
            f.flush() # 实时写入文件，防止程序崩溃丢失数据

    print(f"\n评估完成！新结果已追加到 {output_file}")

if __name__ == "__main__":
    evaluate_models()