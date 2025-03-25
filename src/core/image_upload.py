# src/core/image_upload.py
import base64
import json
import numpy as np
from pathlib import Path
from PIL import Image

from src.repo.split_images_repo import save_to_db
from src.utils.data_conversion import numpy_to_base64
from src.core.image_similarity import extract_feature


def upload_splitted_image_to_db(image_data: np.ndarray, splitted_image_id: str, splitted_image_path: str,
                                original_image_id: str, bounding_box: str, image_format: str, vector: str):
    """
    上传切割后的图像数据（Base64 编码）及图像特征到数据库。

    参数:
    - image_data (np.ndarray): 输入图像的 NumPy 数组。
    - splitted_image_id (str): 切割后的图像 ID。
    - splitted_image_path (str): 图像保存路径。
    - original_image_id (str): 原始图像的 ID。
    - bounding_box (str): 目标检测的边界框数据。
    - image_format (str): 图像的格式（如 'PNG'、'JPEG'）。
    - vector (str): 图像特征的形式。
    """
    # 将图像数据转换为 Base64 编码
    base64_image = numpy_to_base64(image_data, image_format)

    binary_data = base64.b64decode(base64_image)

    # 将 NumPy 数组直接转为 JSON 字符串
    vector = json.dumps(vector.tolist())  # 将 NumPy 数组转换为列表并序列化

    save_to_db(splitted_image_id, splitted_image_path, original_image_id, bounding_box, binary_data, vector)


def process_and_upload_image(image_array: np.ndarray, idx: int, save_path: Path, image_name: str):
    """
    处理并上传分割图像到数据库

    参数:
    - image_array (np.ndarray): 图像数组
    - idx (int): 图像索引
    - save_path (Path): 保存路径
    - image_name (str): 原始图像名称
    """
    # 提取图像特征
    vector = extract_feature(image_array)

    # 保存图像到文件系统
    img = Image.fromarray(image_array)
    img.save(save_path)

    # 创建相对路径格式 {original_image_id}/{segment_idx.png}
    relative_path = f"{image_name}/segment_{idx}.png"

    # 创建包含原始图像名称的唯一ID
    unique_splitted_image_id = f"{image_name}_segment_{idx}"

    # 上传图像到数据库
    upload_splitted_image_to_db(
        image_data=image_array,
        splitted_image_id=unique_splitted_image_id,  # 使用包含原始图像名称的唯一ID
        splitted_image_path=relative_path,  # 使用相对路径
        original_image_id=image_name,
        bounding_box=str(idx),
        image_format="png",
        vector=vector
    )

    return save_path