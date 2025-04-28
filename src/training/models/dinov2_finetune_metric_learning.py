import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import json
from pathlib import Path
from PIL import Image
from src.core.image_similarity.image_similarity_DINOv2 import ImageSimilarityDINOv2
from src.training.losses.contrastive_loss import MultiLabelContrastiveLoss
import random
import numpy as np

# 配置日志格式
import logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)


def split_dataset(data_dirs, train_ratio=0.7, val_ratio=0.2):
    """分层抽样确保标签分布均衡（简化实现，可根据实际标签分布扩展）"""
    all_pairs = []
    for data_dir in data_dirs:
        data_path = Path(data_dir)
        pairs = [(p, float(data_dir.split('_')[-1])) for p in data_path.glob("pair_*") if p.is_dir()]
        all_pairs.extend(pairs)

    random.shuffle(all_pairs)
    num_pairs = len(all_pairs)
    num_train = int(num_pairs * train_ratio)
    num_val = int(num_pairs * val_ratio)

    train_pairs = [p[0] for p in all_pairs[:num_train]]
    val_pairs = [p[0] for p in all_pairs[num_train:num_train + num_val]]
    test_pairs = [p[0] for p in all_pairs[num_train + num_val:]]

    return train_pairs, val_pairs, test_pairs


def evaluate_model(model, pairs, device, transform):
    model.eval()
    total_loss = 0
    total_mse = 0
    total_mae = 0
    total_correct = 0
    total_positive = 0
    total_true_positive = 0
    num_pairs = len(pairs)
    criterion = MultiLabelContrastiveLoss()

    with torch.no_grad():
        for pair_dir in pairs:
            try:
                img1 = Image.open(pair_dir / "image_01.jpg").convert("RGB")
                img2 = Image.open(pair_dir / "image_02.jpg").convert("RGB")
                with open(pair_dir / "metadata.json", "r") as f:
                    label = json.load(f)["similarity"]
                    label_tensor = torch.tensor([label], dtype=torch.float32, device=device)

                img1_t = transform(img1).unsqueeze(0).to(device)
                img2_t = transform(img2).unsqueeze(0).to(device)

                feat1 = model(img1_t)
                feat2 = model(img2_t)
                loss = criterion(feat1, feat2, label_tensor)
                cos_sim = nn.functional.cosine_similarity(feat1, feat2, dim=1).item()

                total_loss += loss.item()
                total_mse += (cos_sim - label) ** 2
                total_mae += np.abs(cos_sim - label)

                # 计算准确率
                if (cos_sim >= 0.5 and label >= 0.5) or (cos_sim < 0.5 and label < 0.5):
                    total_correct += 1

                # 计算召回率
                if label == 1:
                    total_positive += 1
                    if cos_sim >= 0.5:
                        total_true_positive += 1

            except Exception as e:
                logging.warning(f"处理 {pair_dir} 失败: {str(e)}")
                num_pairs -= 1
                continue

    if num_pairs == 0:
        return 0, 0, 0, 0, 0

    avg_loss = total_loss / num_pairs
    avg_mse = total_mse / num_pairs
    avg_mae = total_mae / num_pairs
    accuracy = total_correct / num_pairs
    recall = total_true_positive / total_positive if total_positive > 0 else 0

    return avg_loss, avg_mse, avg_mae, accuracy, recall


def finetune_dinov2(train_data_dirs, epochs=20, lr=0.0003, weight_decay=0.02):
    base_model = ImageSimilarityDINOv2().model
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    base_model.to(device)

    # 冻结前 80% 的 Transformer 块
    num_blocks = len(base_model.blocks)
    freeze_blocks = int(num_blocks * 0.8)
    for block in base_model.blocks[:freeze_blocks]:
        for param in block.parameters():
            param.requires_grad = False

    # 解冻顶层块和投影头
    for block in base_model.blocks[freeze_blocks:]:
        for param in block.parameters():
            param.requires_grad = True
    for param in base_model.head.parameters():
        param.requires_grad = True

    # 打印可训练参数数量
    trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    logging.info(f"可训练参数数量: {trainable_params}")

    base_model.train()

    # 损失函数与优化器
    criterion = MultiLabelContrastiveLoss(max_margin=1.5)  # 调整max_margin适配[0,1]标签
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, base_model.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)  # 余弦退火学习率衰减

    # 数据增强
    transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 数据集划分（扩大验证集至20%）
    train_pairs, val_pairs, test_pairs = split_dataset(train_data_dirs, train_ratio=0.7, val_ratio=0.2)
    logging.info(f"数据集划分: 训练集{len(train_pairs)}对, 验证集{len(val_pairs)}对, 测试集{len(test_pairs)}对")

    best_val_mae = float('inf')
    best_val_accuracy = 0
    early_stop_counter = 0
    early_stop_threshold = 3  # 连续3轮验证集指标未改善则停止

    for epoch in range(epochs):
        total_loss = 0
        processed_pairs = 0

        # 训练阶段
        for pair_dir in train_pairs:
            try:
                img1 = Image.open(pair_dir / "image_01.jpg").convert("RGB")
                img2 = Image.open(pair_dir / "image_02.jpg").convert("RGB")
                with open(pair_dir / "metadata.json", "r") as f:
                    label = json.load(f)["similarity"]
                    label_tensor = torch.tensor([label], dtype=torch.float32, device=device)

                img1_t = transform(img1).unsqueeze(0).to(device)
                img2_t = transform(img2).unsqueeze(0).to(device)

                optimizer.zero_grad()
                feat1 = base_model(img1_t)
                feat2 = base_model(img2_t)
                loss = criterion(feat1, feat2, label_tensor)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                processed_pairs += 1

            except Exception as e:
                logging.error(f"训练样本处理失败: {str(e)}", exc_info=False)
                continue

        if processed_pairs == 0:
            logging.error("警告：本轮未处理任何样本，跳过训练")
            continue

        avg_train_loss = total_loss / processed_pairs
        scheduler.step()  # 更新学习率

        # 验证阶段
        val_loss, val_mse, val_mae, val_accuracy, val_recall = evaluate_model(base_model, val_pairs, device, transform)

        # 早停逻辑
        if val_mae < best_val_mae or val_accuracy > best_val_accuracy:
            best_val_mae = val_mae
            best_val_accuracy = val_accuracy
            early_stop_counter = 0
            torch.save(base_model.state_dict(), "models/best_model_1.pth")
            logging.info(f"✅ 保存新最佳模型（验证MAE={best_val_mae:.4f}, 验证准确率={val_accuracy:.4f}）")
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_threshold:
                logging.warning(f"⚠️ 早停触发：验证MAE或准确率连续{early_stop_threshold}轮未改善")
                break

        # 详细日志输出
        logging.info(
            f"Epoch {epoch + 1}/{epochs} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
            f"训练损失: {avg_train_loss:.4f} | "
            f"验证损失: {val_loss:.4f} | "
            f"验证MAE: {val_mae:.4f} | "
            f"验证准确率: {val_accuracy:.4f} | "
            f"验证召回率: {val_recall:.4f} | "
            f"早停计数器: {early_stop_counter}/{early_stop_threshold}"
        )

    # 最终测试
    test_loss, test_mse, test_mae, test_accuracy, test_recall = evaluate_model(base_model, test_pairs, device, transform)
    logging.info(f"\n最终测试结果 | 测试损失: {test_loss:.4f} | 测试MAE: {test_mae:.4f} | 测试准确率: {test_accuracy:.4f} | 测试召回率: {test_recall:.4f}")

    # 保存最终模型
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    torch.save(base_model.state_dict(), models_dir / "dinov2_finetuned_metric_learning.pth")
    logging.info(f"模型已保存至 {models_dir / 'dinov2_finetuned_metric_learning.pth'}")


if __name__ == "__main__":
    train_data_dirs = [
        "/Users/sunyuliang/Desktop/AppBuilder/Python/dinov2_train/Label_0",
        "/Users/sunyuliang/Desktop/AppBuilder/Python/dinov2_train/Label_0.5",
        "/Users/sunyuliang/Desktop/AppBuilder/Python/dinov2_train/Label_0.75",
        "/Users/sunyuliang/Desktop/AppBuilder/Python/dinov2_train/Label_0.9",
        "/Users/sunyuliang/Desktop/AppBuilder/Python/dinov2_train/Label_1"
    ]

    finetune_dinov2(
        train_data_dirs=train_data_dirs,
        epochs=20,  # 允许更多轮次供早停判断
        lr=0.0003,  # 降低初始学习率
        weight_decay=0.02  # 增加正则化强度
    )
