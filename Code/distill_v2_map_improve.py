import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv
from ultralytics.utils.loss import v8DetectionLoss

# --- 1. 蒸馏损失定义 (确保特征对齐) ---
class DistillLoss(v8DetectionLoss):
    def __init__(self, model, teacher_model, distill_weight=1.0):
        super().__init__(model)
        self.teacher_model = teacher_model
        self.distill_weight = distill_weight
        # 教师模型通常输出 3 个尺度的特征图：P3, P4, P5
        # 对应通道数一般为 [256, 512, 1024] (针对v8s)

    def __call__(self, preds, batch):
        # 计算基础 YOLO 损失
        loss = super().__call__(preds, batch)
        total_loss, loss_items = loss[0], loss[1]

        # 教师模型推理
        with torch.no_grad():
            t_preds = self.teacher_model(batch['img'])

        # Logits 蒸馏 (针对 Detect 层输出)
        # 对分类分支进行 KL 散度约束，温度系数 T=2.0
        T = 2.0
        soft_target = torch.softmax(t_preds[1] / T, dim=-1)
        log_preds = torch.log_softmax(preds[1] / T, dim=-1)
        
        distill_loss = nn.functional.kl_div(log_preds, soft_target, reduction='batchmean') * (T ** 2)

        # 最终损失融合
        # 增加总 Loss，促使学生模型强行向教师对齐
        return total_loss + distill_loss * self.distill_weight, loss_items

# --- 2. 运行脚本 ---
if __name__ == "__main__":
    # 路径配置 (使用你原本的 improve 版本 YAML)
    STUDENT_YAML = r"D:\Capstone\Code\ultralytics\ultralytics\cfg\models\v8\fusion\yolov8s_sg_eca_afpn_improve.yaml"
    TEACHER_PT = r"D:\Capstone\runs\detect\yolov8s_mini\weights\best.pt" 
    DATA_YAML = r"D:\Capstone\Code\visdrone_mini.yaml"

    # 初始化学生模型 (加载你修改后的 C2f_SG_ECA 和 ASFF_2L)
    model = YOLO(STUDENT_YAML)
    
    # 初始化教师模型
    teacher = YOLO(TEACHER_PT).model
    teacher.eval()

    # 训练参数 (回归 640/100 节奏)
    train_args = dict(
        data=DATA_YAML,
        epochs=100,
        imgsz=640,
        batch=8,           # width=0.5 参数量小，batch 可以适当开大
        device=0,
        workers=0,
        # --- 针对 VisDrone 的 Recall 优化 ---
        box=12.0,           # 提高回归损失权重
        cls=1.0,
        dfl=1.5,
        # --- 训练策略 ---
        lr0=0.01,
        warmup_epochs=3,
        name='sea_distill_improve_final',
        exist_ok=True
    )

    # 启动训练
    # 注意：如果你之前在 C2f_SG_ECA 里的 e=1.0，
    # 那么即使 width=0.5，通道也会比原生 v8s 稍厚，这有助于提升 mAP
    model.train(**train_args)