import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
import time
import threading
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="xFormers is not available.*")
from .image_similarity import ImageSimilarity  # 导入原类


class ImageSimilarityDINOv2(ImageSimilarity):
    """
    基于DINOv2的图像相似度分析器（含特征归一化与先进度量方法）
    """

    def __init__(self):
        # 加载DINOv2模型并配置设备
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.model.eval()
        self.model.head = torch.nn.Identity()  # 移除分类头
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # 预处理流程（官方标准+抗锯齿）
        self.transform = transforms.Compose([
            transforms.Resize(224, antialias=True),  # 保持图像质量
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        super().__init__()
        self._model_lock = threading.Lock()

    def extract_feature(self, img_input):
        """特征提取+L2归一化（核心优化点）"""
        with self._model_lock:
            img = self._handle_input(img_input)
            img_t = self.transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                feat = self.model(img_t)

            # 关键：对特征向量进行L2归一化（提升距离度量稳定性）
            feat_np = feat.cpu().squeeze(0).numpy()
            return feat_np / np.linalg.norm(feat_np) if np.linalg.norm(feat_np) != 0 else feat_np

    def _handle_input(self, img_input):
        """输入类型处理（保持原逻辑）"""
        if isinstance(img_input, (str, Path)):
            return Image.open(img_input).convert('RGB')
        elif isinstance(img_input, Image.Image):
            return img_input
        elif isinstance(img_input, np.ndarray):
            return Image.fromarray(img_input)
        else:
            raise ValueError("Unsupported input type")

    @staticmethod
    def cosine_similarity(vec1, vec2):
        """余弦相似度归一化到[0,1]"""
        cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return (cos_sim + 1) / 2  # 映射到0-1区间

    @staticmethod
    def euclidean_similarity(vec1, vec2):
        """欧几里得距离+高斯核（自适应带宽）"""
        dist = np.linalg.norm(vec1 - vec2)
        sigma = np.sqrt(vec1.shape[0])  # 带宽为特征维度平方根
        return np.exp(-dist**2 / (2 * sigma**2))  # 输出(0,1]

    @staticmethod
    def manhattan_similarity(vec1, vec2):
        """曼哈顿距离+理论上界归一化"""
        dist = np.sum(np.abs(vec1 - vec2))
        max_dist = 2 * vec1.shape[0]  # L1范数最大可能值（特征归一化后范围[-1,1]）
        return 1 - (dist / max_dist) if max_dist != 0 else 0  # 映射到[0,1]

    def compare_similarities(self, single_dict, images_dict, metric='cosine'):
        """统一相似度计算接口"""
        single_name, single_vec = list(single_dict.items())[0]
        metric_funcs = {
            'cosine': self.cosine_similarity,
            'euclidean': self.euclidean_similarity,
            'manhattan': self.manhattan_similarity
        }

        if metric not in metric_funcs:
            raise ValueError(f"Unsupported metric: {metric}")

        return [(name, metric_funcs[metric](single_vec, vec))
                for name, vec in images_dict.items()]