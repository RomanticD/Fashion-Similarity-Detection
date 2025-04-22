# src/training/models/dinov2_finetune.py
import torch
import torch.nn as nn
from src.core.image_similarity.image_similarity_DINOv2 import ImageSimilarityDINOv2
import logging

logger = logging.getLogger(__name__)


class DINOv2Finetune(ImageSimilarityDINOv2, nn.Module):
    def __init__(self, num_classes=2, freeze_backbone=True):
        """
        DINOv2微调模型初始化
        :param num_classes: 分类任务类别数（默认2分类：相似/不相似）
        :param freeze_backbone: 是否冻结主干网络（默认True）
        """
        super().__init__()  # 先初始化父类的设备和模型加载
        nn.Module.__init__(self)  # 显式初始化nn.Module基类

        # 1. 冻结主干网络（可选）
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            logger.info("DINOv2主干网络已冻结，仅训练分类头")

        # 2. 添加可训练的分类头（基于特征向量维度动态创建）
        feature_dim = self.model.head.in_features  # 获取DINOv2特征维度（默认768）
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        self.classifier.to(self.device)  # 确保分类头在目标设备上

    def forward(self, x):
        """
        前向传播：特征提取+分类头计算
        :param x: 输入图像（PIL.Image/ndarray/文件路径）
        :return: 分类 logits 或 特征向量（根据训练模式）
        """
        # 复用父类的特征提取流程（含预处理和L2归一化）
        feat = super().extract_feature(x)

        # 转换为Tensor并移动到设备（使用nn.Module的to方法）
        feat_tensor = torch.from_numpy(feat).to(self.device)

        # 通过分类头计算输出
        return self.classifier(feat_tensor)

    def train_mode(self, mode=True):
        """切换训练/推理模式（自动处理主干网络和分类头）"""
        self.model.eval()  # 保持主干网络固定（即使在训练时也不启用dropout/bn）
        self.classifier.train(mode)  # 仅分类头参与训练

    def save_pretrained(self, save_path):
        """保存微调后的模型（仅保存分类头和训练配置）"""
        torch.save({
            'classifier_state_dict': self.classifier.state_dict(),
            'freeze_backbone': self.model.parameters().__next__().requires_grad is False,
            'feature_dim': self.model.head.in_features
        }, save_path)

    @classmethod
    def from_pretrained(cls, pretrained_path, num_classes=2):
        """从预训练模型加载（保持主干网络与原类一致）"""
        model = cls(num_classes=num_classes)
        checkpoint = torch.load(pretrained_path)
        model.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        model.model.head.requires_grad = not checkpoint['freeze_backbone']  # 恢复冻结状态
        return model
