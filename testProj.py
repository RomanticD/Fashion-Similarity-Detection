import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import pymysql  # 或其他数据库驱动

# 1) 加载预训练模型
model = models.resnet50(pretrained=True)
model.eval()  # 设置为推理模式

# 去掉最终分类层 (fc) 只保留到 global avg pool
model.fc = torch.nn.Identity()

# 2) 定义预处理Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def extract_feature(img_path):
    """给定图片路径，返回提取的 ResNet50 特征（2048 维向量）"""
    img = Image.open(img_path).convert('RGB')
    img_t = transform(img)
    img_t = img_t.unsqueeze(0)  # 增加batch维度

    with torch.no_grad():
        feat = model(img_t)  # [1, 2048]
    return feat.squeeze(0).numpy()


def cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度"""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


def main():
    # 设置numpy打印选项，threshold=np.inf表示打印时不省略任何值
    # np.set_printoptions(threshold=np.inf)

    # 两张图片路径
    image_path1 = "testCoat.png"
    image_path2 = "testCoat2.png"

    # 提取特征向量
    feature_vector1 = extract_feature(image_path1)
    feature_vector2 = extract_feature(image_path2)

    # 分别打印两张图片的特征维度和向量
    print(f"Feature vector shape of {image_path1}:", feature_vector1.shape)
    print(f"Feature vector ({image_path1}):", feature_vector1)

    print(f"Feature vector shape of {image_path2}:", feature_vector2.shape)
    print(f"Feature vector ({image_path2}):", feature_vector2)

    # 计算并打印两张图片的相似度（余弦相似度）
    similarity = cosine_similarity(feature_vector1, feature_vector2)
    print(f"Cosine similarity between {image_path1} and {image_path2}: {similarity}")


if __name__ == '__main__':
    main()