# src/core/image_similarity/image_similarity_DINOv2.py
import torch
import torchvision.transforms as transforms
import numpy as np
import warnings
from .image_similarity_base import ImageSimilarityBase
import logging
import time

warnings.filterwarnings("ignore", category=UserWarning, message="xFormers is not available.*")
logger = logging.getLogger(__name__)


class ImageSimilarityDINOv2(ImageSimilarityBase):
    def __init__(self):
        """初始化DINOv2模型及预处理流程"""
        super().__init__()  # 调用基类初始化（获取线程锁和日志配置）

        # 加载DINOv2模型并配置设备
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.model.eval()
        self.model.head = torch.nn.Identity()  # 移除分类头
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        logger.info(f"DINOv2模型加载完成，设备: {self.device.type}")

        # 预处理流程（官方标准+抗锯齿+尺寸适配）
        self.transform = transforms.Compose([
            transforms.Resize(224, antialias=True),  # 保持图像质量
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract_feature(self, img_input) -> np.ndarray:
        """
        提取DINOv2特征向量（实现抽象方法）
        :param img_input: 支持文件路径、PIL图像、NumPy数组
        :return: L2归一化后的特征向量
        """
        start_time = time.time()
        try:
            # 统一输入处理（使用基类方法）
            img = self._process_input(img_input)

            # 预处理并转换设备
            img_t = self.transform(img).unsqueeze(0).to(self.device)

            # 线程安全推理
            with self._model_lock:
                with torch.no_grad():
                    feat = self.model(img_t)

            # 转换为numpy并归一化
            feat_np = feat.cpu().squeeze(0).numpy()
            norm = np.linalg.norm(feat_np)
            feat_normalized = feat_np / norm if norm != 0 else feat_np
            logger.info(
                f"DINOv2特征提取完成，耗时: {time.time()-start_time:.4f}s, "
                f"特征维度: {feat_np.shape[0]}, 设备: {self.device.type}"
            )
            return feat_normalized
        except Exception as e:
            logger.error(f"DINOv2特征提取失败: {str(e)}", exc_info=True)
            raise  # 保持异常调用链
