# image_similarity.py
import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
import time
import tempfile

# ---------------------
# 1) 加载预训练的 ResNet50 模型
# ---------------------
weights = ResNet50_Weights.IMAGENET1K_V1
model = models.resnet50(weights=weights)
model.eval()
model.fc = torch.nn.Identity()  # 保留2048维特征

# ---------------------
# 2) 预处理Transforms
# ---------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def extract_feature(img_input):
    """支持多种输入格式的特征提取"""
    start_time = time.time()

    # 处理不同输入类型
    if isinstance(img_input, (str, Path)):  # 文件路径
        img = Image.open(img_input).convert('RGB')
    elif isinstance(img_input, Image.Image):  # PIL图像对象
        img = img_input
    elif isinstance(img_input, np.ndarray):  # numpy数组
        img = Image.fromarray(img_input)
    else:
        raise ValueError("Unsupported input type")

    # 预处理流程
    img_t = transform(img).unsqueeze(0)

    with torch.no_grad():
        feat = model(img_t)

    feature = feat.squeeze(0).numpy()
    print(f"特征提取耗时: {time.time() - start_time:.4f}s")
    return feature


def load_images_from_arrays(image_arrays):
    """从内存中的图像数组加载特征"""
    return {f"segment_{i}": extract_feature(arr) for i, arr in enumerate(image_arrays)}


def load_single_image_feature_vector(img_path):
    """加载单张图片特征"""
    return {Path(img_path).name: extract_feature(img_path)}


def cosine_similarity(vec1, vec2):
    """计算余弦相似度"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def compare_similarities(single_dict, images_dict):
    """对比相似度"""
    single_name, single_vec = list(single_dict.items())[0]
    return [(name, cosine_similarity(single_vec, vec)) for name, vec in images_dict.items()]