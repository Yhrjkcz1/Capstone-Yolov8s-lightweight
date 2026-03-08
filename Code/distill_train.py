import torch
import torch.nn.functional as F
from ultralytics import YOLO

# 1. 定义蒸馏逻辑的函数
def on_train_epoch_start(trainer):
    """在每一轮训练开始前，确保教师模型在正确的设备上"""
    if not hasattr(trainer, 'teacher_model'):
        teacher_weights = r"D:\Capstone\runs\detect\yolov8s\weights\best.pt"
        print(f"--- 挂载教师模型: {teacher_weights} ---")
        # 加载教师模型
        trainer.teacher_model = YOLO(teacher_weights).model
        trainer.teacher_model.eval()
        for p in trainer.teacher_model.parameters():
            p.requires_grad = False
    
    # 将教师模型移动到当前训练设备
    trainer.teacher_model.to(trainer.device)

def custom_criterion_with_distill(trainer, preds, batch):
    """
    替换原始损失计算，注入蒸馏逻辑
    """
    # 计算学生模型原本的损失
    loss, loss_items = trainer.model.criterion(preds, batch)
    
    # 教师模型前向传播
    imgs = batch["img"]
    if trainer.args.half:
        imgs = imgs.half()
        
    with torch.no_grad():
        t_preds = trainer.teacher_model(imgs)

    # 计算蒸馏损失 (MSE 对齐 Detect 层的 Logits)
    # preds[1] 是学生原始输出，t_preds 是教师原始输出
    distill_loss = 0
    for s_logit, t_logit in zip(preds[1], t_preds):
        distill_loss += F.mse_loss(s_logit, t_logit.detach())
    
    # 权重设为 0.4，你可以根据效果调整
    total_loss = loss + (distill_loss * 0.4)
    
    return total_loss, loss_items

# 2. 启动训练
if __name__ == "__main__":
    # 初始化学生模型 (SEA)
    model = YOLO(r"D:\Capstone\Code\ultralytics\ultralytics\cfg\models\v8\fusion\yolov8s_sg_eca_afpn_improve.yaml")

    # 训练参数
    train_args = dict(
        data=r"D:\Capstone\Code\visdrone_mini.yaml",
        epochs=100,
        imgsz=640,
        batch=6,
        device=0,
        half=True,
        cls=1.3,
        warmup_epochs=8,
        cos_lr=True,
        workers=0,
        name='sea_distill_mini'
    )

    # 注入蒸馏钩子
    # 注意：我们直接修改 trainer 对象的属性
    def setup_distill(trainer):
        trainer.criterion = lambda preds, batch: custom_criterion_with_distill(trainer, preds, batch)

    # 绑定回调
    model.add_callback("on_train_start", setup_distill)
    model.add_callback("on_train_epoch_start", on_train_epoch_start)

    # 开始训练
    model.train(**train_args)