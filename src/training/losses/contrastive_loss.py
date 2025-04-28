import torch
import torch.nn as nn


class MultiLabelContrastiveLoss(nn.Module):
    def __init__(self, max_margin=2.0):
        super(MultiLabelContrastiveLoss, self).__init__()
        self.max_margin = max_margin

    def forward(self, output1, output2, label):
        # output1, output2: 两个样本的特征向量（已归一化，余弦相似度范围[-1,1]）
        cos_sim = torch.nn.functional.cosine_similarity(output1, output2, dim=1)
        # 转换为距离：1 - cos_sim（范围[0,2]）
        distance = 1 - cos_sim

        # 根据标签动态调整 margin
        margin = self.max_margin * (1 - label)

        # 对比损失公式
        loss = torch.mean(
            label * torch.square(distance) +  # 相似度高的样本对拉近距离
            (1 - label) * torch.square(torch.clamp(margin - distance, min=0.0))  # 相似度低的样本对推远距离
        )
        return loss
