# 假设这个文件名为 src/core/image_similarity_vit.py
import torch
import torchvision.models as models
from torchvision.models import vit_b_16, ViT_B_16_Weights
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
import time
import tempfile
import threading
from .image_similarity import ImageSimilarity  # 导入原类

class ImageSimilarityViT(ImageSimilarity):
    """
    A class for extracting features from images and comparing their similarities using ViT model.
    """

    def __init__(self):
        """
        Initialize the image similarity analyzer with a pre-trained ViT model.
        """
        # Load pre-trained ViT model
        self.weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.model = vit_b_16(weights=self.weights)
        self.model.eval()
        # 移除最后一层分类器以获取特征向量
        self.model.heads = torch.nn.Identity()

        # 复用原类的预处理步骤
        super().__init__()
        # 重新定义模型锁
        self._model_lock = threading.Lock()

    def extract_feature(self, img_input):
        """
        Extract features from an image.

        Args:
            img_input: Input image (file path, PIL image, or numpy array).

        Returns:
            np.ndarray: The extracted feature vector.

        Raises:
            ValueError: If the input type is not supported.
        """
        start_time = time.time()

        # 复用原类的输入类型处理逻辑
        img = self._handle_input(img_input)

        # 复用原类的预处理逻辑
        img_t = self.transform(img).unsqueeze(0)

        # Use lock to ensure thread safety when using the model
        with self._model_lock:
            with torch.no_grad():
                feat = self.model(img_t)

        feature = feat.squeeze(0).numpy()
        print(f"Feature extraction time: {time.time() - start_time:.4f}s")
        return feature

    def _handle_input(self, img_input):
        """
        Handle different input types.

        Args:
            img_input: Input image (file path, PIL image, or numpy array).

        Returns:
            Image.Image: The input image as a PIL image.

        Raises:
            ValueError: If the input type is not supported.
        """
        if isinstance(img_input, (str, Path)):  # File path
            img = Image.open(img_input).convert('RGB')
        elif isinstance(img_input, Image.Image):  # PIL image
            img = img_input
        elif isinstance(img_input, np.ndarray):  # NumPy array
            img = Image.fromarray(img_input)
        else:
            raise ValueError("Unsupported input type")
        return img
