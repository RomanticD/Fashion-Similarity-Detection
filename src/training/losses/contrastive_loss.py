# src/training/losses/contrastive_loss.py
import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # output1, output2: 两个样本的特征向量（已归一化，余弦相似度范围[-1,1]）
        cos_sim = torch.dot(output1, output2)
        # 转换为距离：1 - cos_sim（范围[0,2]）
        distance = 1 - cos_sim
        # 对比损失公式
        loss = torch.mean(
            (1 - label) * torch.square(distance) +  # 正样本对（label=1）拉近距离
            label * torch.square(torch.clamp(self.margin - distance, min=0.0))  # 负样本对（label=0）推远距离
        )
        return loss
