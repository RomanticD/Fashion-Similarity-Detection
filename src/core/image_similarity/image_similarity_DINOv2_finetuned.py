# src/core/image_similarity/image_similarity_DINOv2_finetuned.py
import torch
import torchvision.transforms as transforms
import numpy as np
from .image_similarity_base import ImageSimilarityBase
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class ImageSimilarityDINOv2Finetuned(ImageSimilarityBase):
    # 修改默认模型路径为训练脚本保存的最佳模型路径
    def __init__(self, model_path: str = "models/best_model.pth"):
        """初始化微调后的DINOv2模型及预处理流程"""
        super().__init__()  # 继承基类线程锁和日志功能

        # 加载自定义训练模型
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件未找到: {model_path}")

        # 加载模型结构（需与训练时一致）
        # 注意：训练脚本使用的是孪生网络结构，需要修改调用类的模型加载方式
        from src.training.models.dinov2_finetune_v2 import SiameseDINOv2
        self.model = SiameseDINOv2()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # 加载训练好的参数
        self._load_pretrained_weights()

        # 修改为确定性的预处理流程，移除随机变换
        self.transform = transforms.Compose([
            transforms.Resize(256),  # 统一调整大小
            transforms.CenterCrop(224),  # 中心裁剪，确保一致性
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 保留训练时的数据增强流程用于参考
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.GaussianBlur(kernel_size=3, sigma=0.1)
            ], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        logger.info(f"微调DINOv2模型加载完成，设备: {self.device.type}")

    def _load_pretrained_weights(self):
        """加载训练好的模型参数（支持CPU/GPU迁移）"""
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()  # 切换评估模式
        logger.info(f"成功加载模型参数: {self.model_path}")

    def extract_feature(self, img_input) -> np.ndarray:
        """
        提取微调后的DINOv2特征向量（实现抽象方法）
        :param img_input: 支持文件路径、PIL图像、NumPy数组
        :return: L2归一化后的特征向量
        """
        start_time = time.time()
        try:
            # 输入处理（复用基类方法）
            img = self._process_input(img_input)

            # 预处理并转换设备
            img_t = self.transform(img).unsqueeze(0).to(self.device)

            # 线程安全推理
            with self._model_lock:
                with torch.no_grad():
                    # 孪生网络需要输入两个图像，这里为了兼容性，使用相同图像作为输入
                    outputs = self.model(img_t, img_t)
                    feat = outputs["feat1"]

            # 转换为numpy并归一化（保持与训练一致的特征后处理）
            feat_np = feat.cpu().squeeze(0).numpy()
            norm = np.linalg.norm(feat_np)
            feat_normalized = feat_np / norm if norm != 0 else feat_np
            logger.info(
                f"特征提取完成，耗时: {time.time()-start_time:.4f}s, "
                f"特征维度: {feat_np.shape[0]}, 设备: {self.device.type}"
            )
            return feat_normalized

        except Exception as e:
            logger.error(f"特征提取失败: {str(e)}", exc_info=True)
            raise


# 使用示例
if __name__ == "__main__":
    # 初始化模型（加载训练好的参数）
    model = ImageSimilarityDINOv2Finetuned()

    # 准备测试图像（假设为同款服装图像对）
    img_path1 = "/path/to/query_image.jpg"
    img_path2 = "/path/to/candidate_image.jpg"

    # 提取特征
    feat1 = model.extract_feature(img_path1)
    feat2 = model.extract_feature(img_path2)

    # 计算余弦相似度
    cos_sim = model.cosine_similarity(feat1, feat2)
    print(f"余弦相似度: {cos_sim:.4f}")

    # 批量比较示例
    single_img_feat = {img_path1: feat1}
    candidate_imgs = {
        "/path/to/candidate1.jpg": model.extract_feature("/path/to/candidate1.jpg"),
        "/path/to/candidate2.jpg": model.extract_feature("/path/to/candidate2.jpg")
    }

    results = model.compare_similarities(single_img_feat, candidate_imgs, metric="cosine")
    print("相似度排名:", sorted(results, key=lambda x: x[1], reverse=True))
