import os
import torch
from ultralytics import YOLO
from thop import profile

def evaluate_models():
    model_paths = [
        r"D:\Capstone\runs\detect\yolov8s\weights\best.pt",
        r"D:\Capstone\runs\detect\yolov8s_ca_visdrone (full dataset)2\weights\best.pt",
        r"D:\Capstone\runs\detect\yolov8s_eca_visdrone (full dataset)\weights\best.pt",
        r"D:\Capstone\runs\detect\yolov8s_sg_eca_visdrone (full dataset)\weights\best.pt",
        r"D:\Capstone\runs\detect\sea_final_compressed_full\weights\best.pt",
        r"D:\Capstone\runs\detect\sea_full_distill_v1\weights\best.pt",
        r"D:\Capstone\runs\detect\yolov8s_kd_baseline_v1\weights\best.pt",
    ]

    data_yaml = r"D:\Capstone\Code\visdrone.yaml"
    output_file = "compare_clean.txt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dummy_input = torch.randn(1, 3, 640, 640).to(device)

    with open(output_file, "w", encoding="utf-8") as f:
        for model_path in model_paths:
            if not os.path.exists(model_path):
                print(f"跳过不存在的文件: {model_path}")
                continue

            print(f"\n正在分析模型: {model_path}")

            model = YOLO(model_path)
            model.model.to(device)

            # Params
            total_params = sum(p.numel() for p in model.model.parameters())
            params_m = total_params / 1e6

            # FLOPs（THOP）
            flops, _ = profile(model.model, inputs=(dummy_input,), verbose=False)
            flops_g = flops / 1e9

            # 精度
            results = model.val(data=data_yaml, imgsz=640, verbose=False)

            precision = results.box.p.mean().item()
            recall = results.box.r.mean().item()
            map50 = results.box.map50.mean().item()
            map5095 = results.box.map.mean().item()

            text = (
                f"模型: {model_path}\n"
                f"Params: {params_m:.2f} M\n"
                f"FLOPs:  {flops_g:.2f} GFLOPs\n"
                f"Precision:  {precision:.4f}\n"
                f"Recall:     {recall:.4f}\n"
                f"mAP50:      {map50:.4f}\n"
                f"mAP50-95:   {map5095:.4f}\n"
                f"{'-'*45}\n"
            )

            print(text)
            f.write(text)

    print(f"\n评估完成！结果已保存到 {output_file}")


if __name__ == "__main__":
    evaluate_models()