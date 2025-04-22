# src/training/utils/trainer.py（完整训练脚本）
import torch
import logging
from torch.utils.data import DataLoader
from src.training.models.dinov2_finetune import DINOv2Finetune
from src.training.losses.contrastive_loss import ContrastiveLoss
from src.training.utils.dataset import MultiFormatPairDataset


def train_finetune_model(config: dict):
    """
    完整的DINOv2微调训练流程
    :param config: 训练配置字典（包含超参数和路径）
    """
    # ---------------------
    # 1. 设备配置
    # ---------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = logging.getLogger(__name__)
    logger.info(f"使用设备: {device.type}")

    # ---------------------
    # 2. 模型初始化
    # ---------------------
    model = DINOv2Finetune(
        num_classes=config["num_classes"],
        freeze_backbone=config["freeze_backbone"]
    )
    model = model.to(device)  # 将模型移动到目标设备
    model.train_mode(True)  # 开启训练模式（分类头训练，主干网络冻结）

    # ---------------------
    # 3. 数据加载
    # ---------------------
    train_dataset = MultiFormatPairDataset(
        list_file=config["train_list_file"],
        image_size=config["image_size"]
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=device.type == "cuda"  # GPU训练时启用锁页内存
    )

    # ---------------------
    # 4. 损失函数与优化器
    # ---------------------
    loss_fn = ContrastiveLoss(margin=config["contrastive_margin"])
    # 仅优化分类头参数（自动过滤冻结的主干网络参数）
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )

    # ---------------------
    # 5. 训练循环
    # ---------------------
    for epoch in range(config["num_epochs"]):
        model.train()  # 确保分类头处于训练模式
        epoch_loss = 0.0

        for batch_idx, ((img1, img2), labels) in enumerate(train_loader):
            # 数据移动到设备
            img1 = img1.to(device, non_blocking=True)
            img2 = img2.to(device, non_blocking=True)
            labels = torch.tensor(labels, dtype=torch.float32, device=device)

            # 前向传播
            with torch.cuda.amp.autocast(enabled=config["use_amp"]):  # 混合精度训练（可选）
                feat1 = model.extract_feature(img1)  # 复用父类特征提取（含L2归一化）
                feat2 = model.extract_feature(img2)
                loss = loss_fn(feat1, feat2, labels)

            # 反向传播
            optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if batch_idx % config["log_interval"] == 0:
                logger.info(
                    f"Epoch [{epoch}/{config['num_epochs']}] "
                    f"Batch {batch_idx}/{len(train_loader)} "
                    f"Loss: {loss.item():.4f} "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}"
                )

        #  epoch结束时记录平均损失
        logger.info(
            f"Epoch {epoch} 平均损失: {epoch_loss / len(train_loader):.4f}"
        )

        # 保存中间检查点（每5个epoch）
        if (epoch + 1) % config["checkpoint_interval"] == 0:
            model.save_pretrained(
                f"{config['checkpoint_dir']}/epoch_{epoch+1}.pth"
            )

    # ---------------------
    # 6. 保存最终模型
    # ---------------------
    model.save_pretrained(config["final_model_path"])
    logger.info("训练完成，最终模型已保存")


# ---------------------
# 训练配置示例
# ---------------------
DEFAULT_CONFIG = {
    "num_classes": 2,
    "freeze_backbone": True,
    "image_size": 224,
    "train_list_file": "data/finetune_data/train.txt",
    "batch_size": 32,
    "num_workers": 4,
    "contrastive_margin": 0.5,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "num_epochs": 100,
    "log_interval": 10,
    "checkpoint_interval": 5,
    "checkpoint_dir": "models/dinov2_finetuned",
    "final_model_path": "models/dinov2_finetuned/final_model.pth",
    "use_amp": True  # 启用混合精度训练（需GPU支持）
}

if __name__ == "__main__":
    train_finetune_model(DEFAULT_CONFIG)
