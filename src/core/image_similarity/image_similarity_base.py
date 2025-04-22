# src/core/image_similarity/image_similarity_base.py
from pathlib import Path
from PIL import Image
import abc
import numpy as np
import logging
import threading
from typing import Any, Dict, List, Tuple

# 配置日志记录器（建议在项目入口统一配置，此处仅示例）
logger = logging.getLogger(__name__)


class ImageSimilarityBase(metaclass=abc.ABCMeta):
    def __init__(self):
        """初始化线程锁和日志配置"""
        self._model_lock = threading.Lock()  # 线程安全锁（子类共享接口级锁）
        super().__init__()

    @abc.abstractmethod
    def extract_feature(self, img_input: Any) -> np.ndarray:
        """抽象方法：提取图像特征（子类必须实现）"""
        pass

    @staticmethod
    def _process_input(img_input: Any) -> Image.Image:
        """
        统一输入类型处理（核心公共逻辑）
        :param img_input: 支持文件路径(str/Path)、PIL Image、NumPy数组
        :return: PIL Image对象
        :raises ValueError: 不支持的输入类型
        """
        if isinstance(img_input, (str, Path)):
            try:
                return Image.open(img_input).convert('RGB')
            except Exception as e:
                raise ValueError(f"文件路径读取失败: {str(e)}") from e
        elif isinstance(img_input, Image.Image):
            return img_input
        elif isinstance(img_input, np.ndarray):
            try:
                return Image.fromarray(img_input)
            except Exception as e:
                raise ValueError(f"NumPy数组转换失败: {str(e)}") from e
        else:
            raise ValueError(
                f"不支持的输入类型: {type(img_input).__name__}, "
                "仅支持str/Path/PIL.Image/ndarray"
            )

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        余弦相似度计算（保留原始范围[-1,1]）
        :param vec1: 特征向量1
        :param vec2: 特征向量2
        :return: 余弦相似度值
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            logger.warning("检测到零范数向量，返回0相似度")
            return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)

    @staticmethod
    def euclidean_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        欧几里得距离相似度（高斯核映射到(0,1]）
        :param vec1: 特征向量1
        :param vec2: 特征向量2
        :return: 相似度值
        """
        dist = np.linalg.norm(vec1 - vec2)
        sigma = np.sqrt(vec1.shape[0]) if vec1.shape[0] != 0 else 1.0  # 避免零维度
        return np.exp(-dist**2 / (2 * sigma**2)) if sigma != 0 else 0.0

    @staticmethod
    def manhattan_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        曼哈顿距离相似度（归一化到[0,1]）
        :param vec1: 特征向量1
        :param vec2: 特征向量2
        :return: 相似度值
        """
        dist = np.sum(np.abs(vec1 - vec2))
        max_dist = 2 * vec1.shape[0] if vec1.shape[0] != 0 else 1.0  # 避免零维度
        return 1 - (dist / max_dist) if max_dist != 0 else 0.0

    def compare_similarities(
            self,
            single_dict: Dict[str, np.ndarray],
            images_dict: Dict[str, np.ndarray],
            metric: str = 'cosine'
    ) -> List[Tuple[str, float]]:
        """
        统一相似度计算接口
        :param single_dict: 单张图像特征字典 {image_name: feature_vector}
        :param images_dict: 多张图像特征字典 {image_name: feature_vector}
        :param metric: 度量方法（cosine/euclidean/manhattan）
        :return: 相似度列表 [(image_name, score)]
        :raises ValueError: 不支持的度量方法
        """
        if not single_dict or not images_dict:
            raise ValueError("输入字典不能为空")

        metric_funcs = {
            'cosine': self.cosine_similarity,
            'euclidean': self.euclidean_similarity,
            'manhattan': self.manhattan_similarity
        }
        if metric not in metric_funcs:
            raise ValueError(f"不支持的度量方法: {metric}, 仅支持{list(metric_funcs.keys())}")

        single_name, single_vec = next(iter(single_dict.items()))
        results = []
        for name, vec in images_dict.items():
            try:
                score = metric_funcs[metric](single_vec, vec)
                results.append((name, float(score)))  # 转换为JSON可序列化的float
            except Exception as e:
                logger.error(f"计算{name}相似度时出错: {str(e)}", exc_info=True)
                results.append((name, 0.0))  # 错误时返回0相似度
        return results
