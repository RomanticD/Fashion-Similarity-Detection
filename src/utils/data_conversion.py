import logging

import base64
from io import BytesIO

import numpy as np
from PIL import Image
import os

from flask import Flask, request, jsonify

# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


def base64_to_numpy(base64_string):
    # 去除可能存在的前缀部分（如：'data:image/png;base64,'）
    if base64_string.startswith('data:image'):
        base64_string = base64_string.split(",", 1)[1]

    # 解码 Base64 数据
    img_data = base64.b64decode(base64_string)

    # 将字节数据转换为图像
    image = Image.open(BytesIO(img_data))
    
    # 确保图像有正确的色彩模式
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # 将图像转换为 NumPy 数组
    image_np = np.array(image)

    return image_np


def numpy_to_base64(image_data: np.ndarray, image_format: str) -> str:
    """
    将图像数据（NumPy 数组）转换为 Base64 编码的字符串。

    参数:
    - image_data (np.ndarray): 输入图像的 NumPy 数组。
    - image_format (str): 图像的格式，例如 'PNG', 'JPEG'。

    返回:
    - str: Base64 编码的图像字符串。
    """
    # 确保图像格式大小写正确
    image_format = image_format.upper()
    
    # 检查图像数据类型，确保能正确保存
    if image_data.dtype != np.uint8:
        logger.warning(f"图像数据类型非标准: {image_data.dtype}，尝试转换为uint8")
        # 如果是浮点型，可能需要进行归一化和转换
        if np.issubdtype(image_data.dtype, np.floating):
            image_data = (image_data * 255).astype(np.uint8)
        else:
            image_data = image_data.astype(np.uint8)
    
    # 检查透明通道处理
    if len(image_data.shape) == 3 and image_data.shape[2] == 4 and image_format != 'PNG':
        logger.info("检测到带透明通道的图像，转换为RGB模式")
        # 如果不是PNG但有透明通道，转为RGB
        img = Image.fromarray(image_data)
        img = img.convert('RGB')
        image_data = np.array(img)
    
    # 创建PIL图像并保存到BytesIO
    img = Image.fromarray(image_data)
    img_byte_arr = BytesIO()
    
    try:
        # PNG格式需要特别处理
        if image_format == 'PNG':
            img.save(img_byte_arr, format='PNG')
        else:
            img.save(img_byte_arr, format=image_format)
        img_byte_arr = img_byte_arr.getvalue()
        
        # 将字节数据转换为 Base64 编码的字符串
        base64_str = base64.b64encode(img_byte_arr).decode('utf-8')
        return base64_str
    except Exception as e:
        logger.error(f"保存图像失败: {e}, 格式: {image_format}")
        # 尝试使用默认格式
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        base64_str = base64.b64encode(img_byte_arr).decode('utf-8')
        return base64_str