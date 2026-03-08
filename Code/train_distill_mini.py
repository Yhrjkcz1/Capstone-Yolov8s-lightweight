import torch
import torch.nn.functional as F
from ultralytics import YOLO

# ------------------------
# 1. 加载 Teacher & Student
# ------------------------
teacher = YOLO("D:/Capstone/runs/detect/yolov8s_mini/weights/best.pt")
student = YOLO("D:/Capstone/Code/ultralytics/ultralytics/cfg/models/v8/fusion/yolov8s_sg_eca_afpn_improve.yaml")

teacher.model.eval()
for p in teacher.model.parameters():
    p.requires_grad = False

# ------------------------
# 2. 训练参数
# ------------------------
lambda_distill = 0.3
device = "cuda:0"

teacher.model.to(device)
student.model.to(device)

trainer = student.trainer  # 复用 Ultralytics Trainer
dataloader = trainer.dataloader
optimizer = trainer.optimizer

# ------------------------
# 3. 蒸馏训练循环
# ------------------------
for epoch in range(100):
    for batch in dataloader:
        imgs = batch["img"].to(device)

        # ---- Teacher forward ----
        with torch.no_grad():
            t_out = teacher.model(imgs)
            t_cls = t_out[1] if isinstance(t_out, tuple) else t_out

        # ---- Student forward ----
        s_out = student.model(imgs)
        s_cls = s_out[1] if isinstance(s_out, tuple) else s_out

        # ---- 原 YOLO loss ----
        loss_yolo = trainer.loss(s_out, batch)

        # ---- 分类蒸馏 loss（关键）----
        loss_distill = F.mse_loss(s_cls, t_cls.detach())

        # ---- 总 loss ----
        loss = loss_yolo + lambda_distill * loss_distill

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}: loss={loss.item():.4f}")