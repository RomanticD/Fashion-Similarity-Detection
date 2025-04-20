import torch
import torchvision.models as models
from torchvision.models import vit_b_16, ViT_B_16_Weights
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
import time
import threading


class ImageSimilarityViT:
    def __init__(self):
        # 加载预训练模型
        self.weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.model = vit_b_16(weights=self.weights)
        self.model.eval()
        self.model.heads = torch.nn.Identity()  # 移除分类头以获取特征向量

        # 预处理流程（保留ViT标准预处理）
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self._model_lock = threading.Lock()

    def extract_feature(self, img_input):
        """特征提取+L2归一化"""
        with self._model_lock:
            img = self._handle_input(img_input)
            img_t = self.transform(img).unsqueeze(0)
            with torch.no_grad():
                feat = self.model(img_t).squeeze(0).numpy()
            # L2归一化确保特征向量范数为1（提升距离度量稳定性）
            return feat / np.linalg.norm(feat) if np.linalg.norm(feat) != 0 else feat

    def _handle_input(self, img_input):
        """输入类型处理（保留原逻辑）"""
        if isinstance(img_input, (str, Path)):
            img = Image.open(img_input).convert('RGB')
        elif isinstance(img_input, Image.Image):
            img = img_input
        elif isinstance(img_input, np.ndarray):
            img = Image.fromarray(img_input)
        else:
            raise ValueError("Unsupported input type")
        return img

    @staticmethod
    def cosine_similarity(vec1, vec2):
        """余弦相似度归一化到[0,1]"""
        return (np.dot(vec1, vec2) + 1) / 2  # 原范围[-1,1]映射到[0,1]

    @staticmethod
    def euclidean_similarity(vec1, vec2):
        """欧几里得距离+高斯核"""
        dist = np.linalg.norm(vec1 - vec2)
        sigma = np.sqrt(vec1.shape[0])  # 自适应带宽（特征维度平方根）
        return np.exp(-dist**2 / (2 * sigma**2))  # 映射到(0,1]

    @staticmethod
    def manhattan_similarity(vec1, vec2):
        """曼哈顿距离+理论上界归一化"""
        dist = np.sum(np.abs(vec1 - vec2))
        max_dist = 2 * vec1.shape[0]  # 归一化上界（L1范数最大可能值）
        return 1 - (dist / max_dist) if max_dist != 0 else 0  # 映射到[0,1]

    def compare_similarities(self, single_dict, images_dict, metric='cosine'):
        """相似度计算主逻辑"""
        single_name, single_vec = list(single_dict.items())[0]

        # 度量方法映射
        metric_funcs = {
            'cosine': self.cosine_similarity,
            'euclidean': self.euclidean_similarity,
            'manhattan': self.manhattan_similarity
        }
        if metric not in metric_funcs:
            raise ValueError(f"Unsupported metric: {metric}")

        return [(name, metric_funcs[metric](single_vec, vec))
                for name, vec in images_dict.items()]
