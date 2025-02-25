
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

    # 将图像转换为 NumPy 数组
    image_np = np.array(image)

    return image_np