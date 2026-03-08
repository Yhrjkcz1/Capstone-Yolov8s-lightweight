import torch
import time
import os
from thop import profile
from ultralytics import YOLO

def evaluate_models():

    model_paths = [
        r"D:\Capstone\runs\detect\yolov8n\weights\last.pt",
        r"D:\Capstone\runs\detect\yolov8s\weights\last.pt",
        r"D:\Capstone\runs\detect\yolov8n_mini\weights\last.pt",
        r"D:\Capstone\runs\detect\yolov8s_mini\weights\last.pt",
        r"D:\Capstone\runs\detect\yolov8s_mini_origin_ghost\weights\last.pt",
        r"D:\Capstone\runs\detect\yolov8s_mini_sg\weights\last.pt",
        r"D:\Capstone\runs\detect\yolov8s_mini_ca\weights\last.pt",
        r"D:\Capstone\runs\detect\yolov8s_mini_eca\weights\last.pt",
        r"D:\Capstone\runs\detect\yolov8s_mini_fusionCa\weights\last.pt",
        r"D:\Capstone\runs\detect\yolov8s_mini_fusionECA\weights\last.pt",
        r"D:\Capstone\runs\detect\yolov8s_mini_fusionSGECA\weights\last.pt",
        r"D:\Capstone\runs\detect\v8s_afpn_precision\weights\last.pt",
        r"D:\Capstone\runs\detect\v8s_afpn_full_100e_fair\weights\last.pt",
    ]

    data_yaml = r"D:\Capstone\Code\visdrone_mini.yaml"
    output_file = "compare_mini_last.txt"

    input_size = (1, 3, 640, 640)
    test_iters = 50

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dummy_input = torch.randn(input_size).to(device)

    # 读取已存在模型记录
    evaluated_models = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("模型:"):
                    evaluated_models.add(line.strip().replace("模型: ", ""))

    with open(output_file, "a", encoding="utf-8") as f:

        for model_path in model_paths:

            if model_path in evaluated_models:
                print(f"跳过已评估模型: {model_path}")
                continue

            print(f"\n分析模型: {model_path}")

            model = YOLO(model_path)
            model.to(device)

            # 参数量
            total_params = sum(p.numel() for p in model.model.parameters())
            params_m = total_params / 1e6

            # FLOPs
            flops, _ = profile(model.model, inputs=(dummy_input,), verbose=False)
            flops_g = flops / 1e9

            # FPS
            model.model.eval()
            if device == "cuda":
                torch.cuda.synchronize()

            start = time.time()
            for _ in range(test_iters):
                _ = model.model(dummy_input)

            if device == "cuda":
                torch.cuda.synchronize()

            fps = test_iters / (time.time() - start)

            # 只对新模型跑验证
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

    print(f"\n完成！结果已追加到 {output_file}")


if __name__ == "__main__":
    evaluate_models()
