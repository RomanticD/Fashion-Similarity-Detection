# src/core/image_similarity/image_similarity_vit.py
import torch
from torchvision.models import vit_b_16, ViT_B_16_Weights
import torchvision.transforms as transforms
import numpy as np
from .image_similarity_base import ImageSimilarityBase
import logging
import time

logger = logging.getLogger(__name__)


class ImageSimilarityViT(ImageSimilarityBase):
    def __init__(self):
        """初始化ViT模型及预处理流程"""
        super().__init__()  # 调用基类初始化（获取线程锁和日志配置）

        # 加载预训练模型
        self.weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.model = vit_b_16(weights=self.weights)
        self.model.eval()
        self.model.heads = torch.nn.Identity()  # 移除分类头以获取特征向量
        logger.info("ViT模型加载完成，使用IMAGENET1K_V1权重")

        # 预处理流程（ViT标准预处理，含抗锯齿优化）
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),  # 保持图像质量
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def extract_feature(self, img_input) -> np.ndarray:
        """
        提取ViT特征向量（实现抽象方法）
        :param img_input: 支持文件路径、PIL图像、NumPy数组
        :return: L2归一化后的特征向量
        """
        start_time = time.time()
        try:
            # 统一输入处理（使用基类方法）
            img = self._process_input(img_input)

            # 预处理
            img_t = self.transform(img).unsqueeze(0)

            # 线程安全推理
            with self._model_lock:
                with torch.no_grad():
                    feat = self.model(img_t).squeeze(0).numpy()

            # L2归一化（提升距离度量稳定性）
            norm = np.linalg.norm(feat)
            feat_normalized = feat / norm if norm != 0 else feat
            logger.info(
                f"ViT特征提取完成，耗时: {time.time()-start_time:.4f}s, "
                f"特征维度: {feat.shape[0]}, 归一化范数: {norm:.4f}"
            )
            return feat_normalized
        except Exception as e:
            logger.error(f"ViT特征提取失败: {str(e)}", exc_info=True)
            raise  # 保持异常调用链
