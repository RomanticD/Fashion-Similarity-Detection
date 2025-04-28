import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from pathlib import Path
from PIL import Image
import random
from collections import defaultdict
import logging
import torch.nn.functional as F
import warnings
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore", category=UserWarning, message="xFormers is not available.*")

# 配置日志格式
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)


def split_dataset(data_dirs, train_ratio=0.7, val_ratio=0.2):
    """分层抽样，确保各标签样本比例均衡"""
    label_groups = defaultdict(list)
    for data_dir in data_dirs:
        label = float(data_dir.split('_')[-1])
        data_path = Path(data_dir)
        pairs = [(p, label) for p in data_path.glob("pair_*") if p.is_dir()]
        label_groups[label].extend(pairs)

    # 分层抽样并保持成对关系
    train_pairs, val_pairs, test_pairs = [], [], []
    for label, paths in label_groups.items():
        if label == 0:
            paths *= 2  # 过采样负样本
        random.shuffle(paths)
        total = len(paths)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        train_pairs.extend(paths[:train_end])
        val_pairs.extend(paths[train_end:val_end])
        test_pairs.extend(paths[val_end:])

    logging.info(f"样本分布 - 训练集: {len(train_pairs)}, 验证集: {len(val_pairs)}, 测试集: {len(test_pairs)}")
    return train_pairs, val_pairs, test_pairs


class SiameseDINOv2(nn.Module):
    """孪生网络架构，支持成对样本特征提取"""
    def __init__(self):
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.backbone.head = nn.Identity()  # 移除分类头

        # 冻结所有层
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 解冻更多层，例如最后十层
        layers_to_unfreeze = list(self.backbone.parameters())[-5:]
        for param in layers_to_unfreeze:
            param.requires_grad = True

        # 添加更多全连接层
        self.fc1 = nn.Linear(768, 512)
        self.relu = nn.ReLU()
        self.cloth_classifier = nn.Linear(512, 1)  # 衣物分类头
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x1, x2):
        # 提取特征并归一化
        feat1 = F.normalize(self.backbone(x1), dim=1)
        feat2 = F.normalize(self.backbone(x2), dim=1)

        # 通过新增的全连接层
        feat1 = self.relu(self.fc1(feat1))
        feat2 = self.relu(self.fc1(feat2))

        # 分类判断（单个样本是否为衣物）
        is_cloth1 = self.cloth_classifier(feat1)
        is_cloth2 = self.cloth_classifier(feat2)
        # 计算余弦相似度（移除keepdim参数，手动调整维度）
        cos_sim = F.cosine_similarity(feat1, feat2, dim=1).unsqueeze(1)  # 手动添加维度
        return {
            "feat1": feat1,
            "feat2": feat2,
            "similarity": cos_sim,
            "is_cloth1": is_cloth1,
            "is_cloth2": is_cloth2
        }


class PairwiseLoss(nn.Module):
    """成对样本损失函数（相似度回归+跨类别分类）"""
    def __init__(self, sim_weight=0.8, class_weight=0.2):
        super().__init__()
        self.sim_weight = sim_weight
        self.class_weight = class_weight
        self.sim_loss_fn = nn.MSELoss()  # 相似度回归损失
        self.class_loss_fn = nn.BCEWithLogitsLoss()  # 跨类别分类损失

    def forward(self, outputs, label):
        # 相似度损失（仅处理label>0的样本）
        sim_mask = label > 0
        valid_sim = outputs["similarity"][sim_mask]
        valid_label = label[sim_mask].view(-1, 1)
        if valid_sim.numel() > 0:
            # 确保 valid_sim 和 valid_label 尺寸一致
            if valid_sim.shape != valid_label.shape:
                valid_sim = valid_sim.view(valid_label.shape)
            sim_loss = self.sim_loss_fn(valid_sim, valid_label)
        else:
            sim_loss = 0.0

        # 跨类别分类损失（label=0为跨类别，label>0为同类别）
        class_label = (label > 0).float().view(-1, 1)  # 1表示同类别（均为衣物），0表示跨类别
        class_logits = torch.cat([outputs["is_cloth1"], outputs["is_cloth2"]], dim=1).mean(dim=1, keepdim=True)
        class_loss = self.class_loss_fn(class_logits, class_label)

        return self.sim_weight * sim_loss + self.class_weight * class_loss


def evaluate_model(model, pairs, device, transform):
    model.eval()
    total_mae = 0.0
    total_class_correct = 0
    total_neg = 0
    total_neg_correct = 0
    num_pairs = len(pairs)

    with torch.no_grad():
        for pair_path, label in pairs:
            try:
                # 读取成对图片
                img1 = Image.open(pair_path / "image_01.jpg").convert("RGB")
                img2 = Image.open(pair_path / "image_02.jpg").convert("RGB")
                # 预处理
                img1_t = transform(img1).unsqueeze(0).to(device)
                img2_t = transform(img2).unsqueeze(0).to(device)
                # 前向传播
                outputs = model(img1_t, img2_t)
                # 相似度评估
                sim_pred = outputs["similarity"].item()
                total_mae += abs(sim_pred - label)
                # 跨类别分类评估（label=0为负样本，否则为正样本）
                class_pred = (torch.sigmoid(outputs["is_cloth1"]) * torch.sigmoid(outputs["is_cloth2"])).item() > 0.5
                class_true = (label > 0)
                if class_pred == class_true:
                    total_class_correct += 1
                # 特异性计算（仅label=0的样本）
                if label == 0:
                    total_neg += 1
                    if not class_pred:
                        total_neg_correct += 1
            except Exception as e:
                logging.warning(f"处理样本失败: {pair_path}, 错误: {str(e)}")
                num_pairs -= 1
                continue

    if num_pairs == 0:
        return 0, 0, 0, 0

    avg_mae = total_mae / num_pairs
    class_accuracy = total_class_correct / num_pairs
    specificity = total_neg_correct / total_neg if total_neg > 0 else 1.0
    return avg_mae, class_accuracy, specificity, num_pairs


class PairDataset(Dataset):
    def __init__(self, pairs, transform):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair_path, label = self.pairs[idx]
        img1 = Image.open(pair_path / "image_01.jpg").convert("RGB")
        img2 = Image.open(pair_path / "image_02.jpg").convert("RGB")
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        return img1, img2, torch.tensor(label, dtype=torch.float32)


def finetune_dinov2(train_data_dirs, epochs=30, lr=1e-4):
    model = SiameseDINOv2()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.001)
    criterion = PairwiseLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 数据增强策略（保留衣物特征的同时增加多样性）
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=0.1)
        ], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载并划分数据集
    train_pairs, val_pairs, test_pairs = split_dataset(train_data_dirs)
    best_val_loss = float('inf')
    best_val_mae = float('inf')
    best_val_acc = 0.0
    best_specificity = 0.0
    early_stop_counter = 0

    train_dataset = PairDataset(train_pairs, transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

    use_cuda = torch.cuda.is_available()
    scaler = torch.amp.GradScaler() if use_cuda else None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = len(train_loader)

        logging.info(f"Epoch {epoch + 1}/{epochs}: 开始训练")
        for batch_idx, (x1, x2, labels) in enumerate(train_loader):
            try:
                x1 = x1.to(model.device)
                x2 = x2.to(model.device)
                labels = labels.view(-1, 1).to(model.device)

                if use_cuda:
                    with torch.amp.autocast(device_type="cuda"):
                        outputs = model(x1, x2)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(x1, x2)
                    loss = criterion(outputs, labels)

                optimizer.zero_grad()
                if use_cuda:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()
                # 输出当前批次的损失
                logging.info(f"Epoch {epoch + 1}/{epochs} - 批次 {batch_idx + 1}/{num_batches} - 批次损失: {loss.item():.4f}")

                # 每10个批次输出一次训练进度
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                    progress = (batch_idx + 1) / num_batches * 100
                    avg_batch_loss = total_loss / (batch_idx + 1)
                    logging.info(f"Epoch {epoch + 1}/{epochs} - 训练进度: {progress:.1f}% - 当前平均批次损失: {avg_batch_loss:.4f}")
            except Exception as e:
                logging.error(f"Epoch {epoch + 1} 批次 {batch_idx + 1} 训练出错: {str(e)}")

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # 验证阶段
        model.eval()
        logging.info(f"Epoch {epoch + 1}/{epochs}: 开始验证")
        val_mae, val_acc, val_spec, valid_pairs_num = evaluate_model(model, val_pairs, model.device, transform)

        # 输出验证阶段的详细信息
        logging.info(f"Epoch {epoch + 1}/{epochs} - 验证阶段 - 有效样本数: {valid_pairs_num}")
        logging.info(f"Epoch {epoch + 1}/{epochs} - 验证阶段 - MAE: {val_mae:.4f}")
        logging.info(f"Epoch {epoch + 1}/{epochs} - 验证阶段 - 分类准确率: {val_acc:.4f}")
        logging.info(f"Epoch {epoch + 1}/{epochs} - 验证阶段 - 特异性: {val_spec:.4f}")

        # 检查是否有指标提升
        improved = False
        if val_mae < best_val_mae or val_acc > best_val_acc or val_spec > best_specificity:
            improved = True
            best_val_mae = min(val_mae, best_val_mae)
            best_val_acc = max(val_acc, best_val_acc)
            best_specificity = max(val_spec, best_specificity)
            torch.save(model.state_dict(), "models/best_model.pth")
            early_stop_counter = 0
            logging.info(f"Epoch {epoch + 1} - 保存最佳模型（MAE={val_mae:.4f}, 分类准确率={val_acc:.4f}, 特异性={val_spec:.4f}）")
        else:
            early_stop_counter += 1

        if early_stop_counter >= 5:
            logging.warning("早停触发：多个指标连续5轮未提升")
            break

        logging.info(f"Epoch {epoch + 1}/{epochs} | 损失: {avg_loss:.4f} | MAE: {val_mae:.4f} | 分类准确率: {val_acc:.4f} | 特异性: {val_spec:.4f}")

    # 最终测试
    model.load_state_dict(torch.load("models/best_model.pth"))
    logging.info("开始最终测试")
    test_mae, test_acc, test_spec, test_pairs_num = evaluate_model(model, test_pairs, model.device, transform)
    logging.info(f"最终测试 - 有效样本数: {test_pairs_num}")
    logging.info(f"最终测试结果 | MAE: {test_mae:.4f} | 分类准确率: {test_acc:.4f} | 特异性: {test_spec:.4f}")
    torch.save(model.state_dict(), "models/siamese_dinov2.pth")


if __name__ == "__main__":
    train_data_dirs = [
        "/Users/sunyuliang/Desktop/AppBuilder/Python/dinov2_train/Label_0",
        "/Users/sunyuliang/Desktop/AppBuilder/Python/dinov2_train/Label_0.5",
        "/Users/sunyuliang/Desktop/AppBuilder/Python/dinov2_train/Label_0.75",
        "/Users/sunyuliang/Desktop/AppBuilder/Python/dinov2_train/Label_0.9",
        "/Users/sunyuliang/Desktop/AppBuilder/Python/dinov2_train/Label_1"
    ]
    finetune_dinov2(train_data_dirs, epochs=30, lr=1e-4)
    