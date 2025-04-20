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
        # Load pre-trained ViT model
        self.weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.model = vit_b_16(weights=self.weights)
        self.model.eval()
        # 移除最后一层分类器以获取特征向量
        self.model.heads = torch.nn.Identity()

        # 调整预处理逻辑
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # ViT模型通常使用224x224尺寸
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],  # 根据ViT模型的要求调整归一化参数
                std=[0.5, 0.5, 0.5]
            )
        ])

        # 重新定义模型锁
        self._model_lock = threading.Lock()

    def extract_feature(self, img_input):
        start_time = time.time()

        # 处理不同的输入类型
        img = self._handle_input(img_input)

        # 预处理图像
        img_t = self.transform(img).unsqueeze(0)

        # 使用锁确保线程安全
        with self._model_lock:
            with torch.no_grad():
                # 前向传播
                outputs = self.model(img_t)
                # 可以考虑提取注意力层的输出，这里简单返回最终的特征向量
                feat = outputs

        feature = feat.squeeze(0).numpy()
        print(f"Feature extraction time: {time.time() - start_time:.4f}s")
        return feature

    def _handle_input(self, img_input):
        if isinstance(img_input, (str, Path)):  # 文件路径
            img = Image.open(img_input).convert('RGB')
        elif isinstance(img_input, Image.Image):  # PIL图像
            img = img_input
        elif isinstance(img_input, np.ndarray):  # NumPy数组
            img = Image.fromarray(img_input)
        else:
            raise ValueError("Unsupported input type")
        return img

    @staticmethod
    def euclidean_distance(vec1, vec2):
        distance = np.linalg.norm(vec1 - vec2)
        return 1 / (1 + distance)

    @staticmethod
    def manhattan_distance(vec1, vec2):
        distance = np.sum(np.abs(vec1 - vec2))
        return 1 / (1 + distance)

    def compare_similarities(self, single_dict, images_dict, metric='cosine'):
        single_name, single_vec = list(single_dict.items())[0]
        if metric == 'cosine':
            similarity_func = self.cosine_similarity
        elif metric == 'euclidean':
            similarity_func = self.euclidean_distance
        elif metric == 'manhattan':
            similarity_func = self.manhattan_distance
        else:
            raise ValueError("Unsupported similarity metric")

        return [(name, similarity_func(single_vec, vec)) for name, vec in images_dict.items()]

    @staticmethod
    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
