import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import json
from pathlib import Path
from PIL import Image
from src.core.image_similarity.image_similarity_DINOv2 import ImageSimilarityDINOv2
import random
import numpy as np


def split_dataset(data_dirs, train_ratio=0.8, val_ratio=0.1):
    train_pairs = []
    val_pairs = []
    test_pairs = []

    for data_dir in data_dirs:
        data_path = Path(data_dir)
        all_pairs = [p for p in data_path.glob("pair_*") if p.is_dir()]
        random.shuffle(all_pairs)

        num_pairs = len(all_pairs)
        num_train = int(num_pairs * train_ratio)
        num_val = int(num_pairs * val_ratio)

        train_pairs.extend(all_pairs[:num_train])
        val_pairs.extend(all_pairs[num_train:num_train + num_val])
        test_pairs.extend(all_pairs[num_train + num_val:])

    return train_pairs, val_pairs, test_pairs


def evaluate_model(model, pairs, device, transform):
    model.eval()
    total_loss = 0
    total_mse = 0
    total_mae = 0
    num_pairs = len(pairs)

    with torch.no_grad():
        for pair_dir in pairs:
            try:
                img1 = Image.open(pair_dir / "image_01.jpg").convert("RGB")
                img2 = Image.open(pair_dir / "image_02.jpg").convert("RGB")
                with open(pair_dir / "metadata.json", "r") as f:
                    label = json.load(f)["similarity"]

                img1_t = transform(img1).unsqueeze(0).to(device)
                img2_t = transform(img2).unsqueeze(0).to(device)

                feat1 = model(img1_t)
                feat2 = model(img2_t)
                cos_sim = nn.functional.cosine_similarity(feat1, feat2, dim=1).item()

                loss = (cos_sim - label) ** 2
                total_loss += loss
                total_mse += loss
                total_mae += np.abs(cos_sim - label)

            except Exception as e:
                print(f"处理 {pair_dir} 时出错: {str(e)}")
                num_pairs -= 1

    if num_pairs == 0:
        return 0, 0, 0

    avg_loss = total_loss / num_pairs
    avg_mse = total_mse / num_pairs
    avg_mae = total_mae / num_pairs

    return avg_loss, avg_mse, avg_mae


def finetune_dinov2(train_data_dirs, epochs=10, lr=0.001):  # 调整 epochs 和 lr
    base_model = ImageSimilarityDINOv2().model
    # 修改设备为 MPS（Apple 芯片专用）
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    base_model.to(device)
    base_model.train()
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(base_model.parameters(), lr=lr, weight_decay=0.01)
    transform = transforms.Compose([
        transforms.Resize(224, antialias=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_pairs, val_pairs, test_pairs = split_dataset(train_data_dirs)

    for epoch in range(epochs):
        total_loss = 0
        processed_pairs = 0

        for pair_dir in train_pairs:
            try:
                img1 = Image.open(pair_dir / "image_01.jpg").convert("RGB")
                img2 = Image.open(pair_dir / "image_02.jpg").convert("RGB")
                with open(pair_dir / "metadata.json", "r") as f:
                    label = json.load(f)["similarity"]

                img1_t = transform(img1).unsqueeze(0).to(device)
                img2_t = transform(img2).unsqueeze(0).to(device)

                optimizer.zero_grad()
                feat1 = base_model(img1_t)
                feat2 = base_model(img2_t)
                cos_sim = nn.functional.cosine_similarity(feat1, feat2, dim=1)
                loss = criterion(cos_sim, torch.tensor([label], dtype=torch.float32).to(device))

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                processed_pairs += 1

            except Exception as e:
                print(f"处理 {pair_dir} 时出错: {str(e)}")
                continue

        if processed_pairs == 0:
            print("警告：未处理任何配对，检查数据集路径与文件完整性")
            continue

        avg_train_loss = total_loss / processed_pairs

        # 在验证集上评估模型
        val_loss, val_mse, val_mae = evaluate_model(base_model, val_pairs, device, transform)

        print(f"Epoch {epoch + 1}/{epochs} - 训练损失: {avg_train_loss:.4f} - 验证损失: {val_loss:.4f} - 验证 MSE: {val_mse:.4f} - 验证 MAE: {val_mae:.4f}")

    # 在测试集上进行最终评估
    test_loss, test_mse, test_mae = evaluate_model(base_model, test_pairs, device, transform)
    print(f"测试损失: {test_loss:.4f} - 测试 MSE: {test_mse:.4f} - 测试 MAE: {test_mae:.4f}")

    # 保存微调后的模型
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    torch.save(base_model.state_dict(), models_dir / "dinov2_finetuned.pth")
    print("训练完成，模型已保存为 models/dinov2_finetuned.pth")


if __name__ == "__main__":
    train_data_dirs = [
        "/Users/sunyuliang/Desktop/AppBuilder/Python/dinov2_train/Label_0",
        "/Users/sunyuliang/Desktop/AppBuilder/Python/dinov2_train/Label_0.5",
        "/Users/sunyuliang/Desktop/AppBuilder/Python/dinov2_train/Label_0.75",
        "/Users/sunyuliang/Desktop/AppBuilder/Python/dinov2_train/Label_0.9",
        "/Users/sunyuliang/Desktop/AppBuilder/Python/dinov2_train/Label_1"
    ]
    finetune_dinov2(train_data_dirs, epochs=10, lr=0.001)
