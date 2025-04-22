# src/core/image_similarity_resnet50.py
import time
import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import torchvision.transforms as transforms
import numpy as np
import logging
from .image_similarity_base import ImageSimilarityBase
from pathlib import Path

logger = logging.getLogger(__name__)


class ImageSimilarityResNet50(ImageSimilarityBase):
    def __init__(self):
        """初始化ResNet50模型和预处理流程"""
        super().__init__()  # 调用父类初始化（获取线程锁）

        # 加载模型
        self.weights = ResNet50_Weights.IMAGENET1K_V1
        self.model = models.resnet50(weights=self.weights)
        self.model.eval()
        self.model.fc = torch.nn.Identity()  # 保留2048维特征

        # 预处理流程
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract_feature(self, img_input) -> np.ndarray:
        """
        提取ResNet50特征向量（实现抽象方法）
        :param img_input: 支持文件路径、PIL图像、NumPy数组
        :return: 2048维特征向量
        """
        start_time = time.time()
        try:
            # 统一输入处理（使用父类方法）
            img = self._process_input(img_input)

            # 预处理
            img_t = self.transform(img).unsqueeze(0)

            # 线程安全推理
            with self._model_lock:
                with torch.no_grad():
                    feat = self.model(img_t)

            feature = feat.squeeze(0).numpy()
            logger.info(
                f"ResNet50特征提取完成，耗时: {time.time()-start_time:.4f}s, "
                f"特征维度: {feature.shape[0]}"
            )
            return feature
        except Exception as e:
            logger.error(f"特征提取失败: {str(e)}", exc_info=True)
            raise  # 向上抛出异常保持调用链

    def load_images_from_arrays(self, image_arrays):
        """
        Load features from in-memory image arrays.

        Args:
            image_arrays (list): A list of numpy arrays representing images.

        Returns:
            dict: A dictionary mapping segment names to feature vectors.
        """
        return {f"segment_{i}": self.extract_feature(arr) for i, arr in enumerate(image_arrays)}

    def load_single_image_feature_vector(self, img_path):
        """
        Load the feature vector of a single image.

        Args:
            img_path (str or Path): The file path of the image.

        Returns:
            dict: A dictionary mapping the image name to its feature vector.
        """
        return {Path(img_path).name: self.extract_feature(img_path)}
