import base64
import json

import numpy as np
from flask import current_app
import os

from src.repo.split_images_repo import save_to_db
from src.utils.data_conversion import numpy_to_base64


def upload_splitted_image_to_db(image_data: np.ndarray, splitted_image_id: str, splitted_image_path: str,
                                 original_image_id: str, bounding_box: str, image_format: str, vector: str, ):
    """
    上传切割后的图像数据（Base64 编码）及图像特征到数据库。

    参数:
    - image_data (np.ndarray): 输入图像的 NumPy 数组。
    - splitted_image_id (str): 切割后的图像 ID。
    - splitted_image_path (str): 图像保存路径。
    - original_image_id (str): 原始图像的 ID。
    - bounding_box (str): 目标检测的边界框数据。
    - image_format (str): 图像的格式（如 'PNG'、'JPEG'）。
    - vector (str): 图像特征的 Base64 编码或其他形式。
    """
    # 将图像数据转换为 Base64 编码
    base64_image = numpy_to_base64(image_data, image_format)

    binary_data = base64.b64decode(base64_image)

    # 将 NumPy 数组直接转为 JSON 字符串
    vector = json.dumps(vector.tolist())  # 将 NumPy 数组转换为列表并序列化

    save_to_db(splitted_image_id, splitted_image_path, original_image_id, bounding_box, binary_data, vector)