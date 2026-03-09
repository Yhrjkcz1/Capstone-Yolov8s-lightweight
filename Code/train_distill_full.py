import torch
import torch.nn as nn
import os
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss

# --- 1. 蒸馏损失定义 ---
class DistillLoss(v8DetectionLoss):
    def __init__(self, model, teacher_model, distill_weight=1.0):
        super().__init__(model)
        self.teacher_model = teacher_model
        self.distill_weight = distill_weight

    def __call__(self, preds, batch):
        # 计算学生模型基础损失 (Box, Class, DFL)
        loss = super().__call__(preds, batch)
        total_loss, loss_items = loss[0], loss[1]

        # 教师模型推理 (eval模式已在初始化设定)
        with torch.no_grad():
            t_preds = self.teacher_model(batch['img'])

        # Logits 蒸馏: 针对分类分支的 KL 散度约束
        # T=2.0 为常用的温度系数，平衡软硬标签
        T = 2.0
        soft_target = torch.softmax(t_preds[1] / T, dim=-1)
        log_preds = torch.log_softmax(preds[1] / T, dim=-1)
        
        # 计算 KL 散度损失
        distill_loss = nn.functional.kl_div(log_preds, soft_target, reduction='batchmean') * (T ** 2)

        # 最终损失融合
        return total_loss + distill_loss * self.distill_weight, loss_items

# --- 2. 运行脚本 ---
if __name__ == "__main__":
    # 【路径配置】确保使用全量数据集 yaml 和之前训练好的最佳教师权重
    STUDENT_YAML = r"D:\Capstone\Code\ultralytics\ultralytics\cfg\models\v8\fusion\yolov8s_sg_eca_afpn_improve.yaml"
    TEACHER_PT = r"D:\Capstone\runs\detect\yolov8s\weights\best.pt" 
    DATA_YAML = r"D:\Capstone\Code\visdrone.yaml" # 全量数据集

    # 路径安全检查：防止自动下载官方权重
    if not os.path.exists(TEACHER_PT):
        raise FileNotFoundError(f"错误：在指定路径未找到教师模型权重: {TEACHER_PT}")

    # A. 初始化学生模型
    # 加载修改后的 C2f_SG_ECA 和 ASFF_2L 代码对应的结构
    model = YOLO(STUDENT_YAML)
    
    # B. 初始化教师模型并冻结
    teacher_model_wrapper = YOLO(TEACHER_PT)
    teacher = teacher_model_wrapper.model
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    # C. 训练参数设置 (针对全量数据集优化)
    train_args = dict(
        data=DATA_YAML,
        epochs=100,
        imgsz=640,
        batch=8,           # 全量训练建议保持 16，如果显存不足再降到 8
        device=0,
        workers=0,          # Windows 环境建议 0
        # --- 针对 VisDrone 小目标的 Recall 补偿 ---
        box=12.0,           
        cls=1.2,            # 略微提高分类权重，全量数据类别分布更广
        dfl=1.5,
        # --- 训练稳定性策略 ---
        lr0=0.01,
        warmup_epochs=3,
        close_mosaic=10,    # 最后 10 轮关闭 mosaic，稳定精准度
        patience=20,        # 20轮内 mAP 不涨则停止，防止过拟合
        name='sea_full_distill_v1',
        exist_ok=True
    )

    # D. 注入自定义蒸馏损失
    # 注意：在 YOLOv8 中，需要覆盖 criterion 属性
    def get_distill_criterion(self):
        return DistillLoss(self.model, teacher, distill_weight=1.0)

    # 强制注入
    model.model.criterion = DistillLoss(model.model, teacher, distill_weight=1.0)

    # E. 启动训练
    model.train(**train_args)