#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试脚本: 进行图片相似度检测测试
用法: python test_image_similarity.py
"""

import argparse
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import base64
import json
import pickle
import shutil
import io

# 导入 ImageSimilarity 类
from src.core.image_similarity import ImageSimilarity
from src.core.image_similarity_vit import ImageSimilarityViT

from src.core.groundingdino_handler import ClothingDetector
from src.repo.split_images_repo import select_image_data_by_id, select_multiple_image_data_by_ids


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='测试图片相似度检测')
    return parser.parse_args()


def find_project_root():
    """查找项目根目录"""
    current_dir = Path.cwd()

    # 尝试向上查找包含README.md的目录
    while current_dir != current_dir.parent:
        if (current_dir / 'README.md').exists():
            return current_dir
        current_dir = current_dir.parent

    # 如果找不到，则使用当前目录
    return Path.cwd()


def load_images(image_dir):
    """加载指定目录下的所有图片"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(image_dir.rglob(f'*{ext}')))  # 使用 rglob 递归查找
        image_files.extend(list(image_dir.rglob(f'*{ext.upper()}')))  # 检查大写扩展名

    if not image_files:
        print(f"❌ 错误: 图片目录中没有找到图片文件: {image_dir}")
        sys.exit(1)

    print(f"✅ 找到 {len(image_files)} 个图片文件")
    return image_files


def image_to_base64(image_path):
    """将图片文件转换为Base64编码字符串"""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    except Exception as e:
        print(f"❌ 错误: 读取图片文件时出错: {e}")
        sys.exit(1)


def load_vector_index(root_dir):
    """加载向量索引"""
    index_file = root_dir / 'vector_nn_index.pkl'
    id_map_file = root_dir / 'vector_id_map.json'
    try:
        with open(index_file, 'rb') as f:
            data = pickle.load(f)
            index = data['index']
            vectors = data['vectors']
        with open(id_map_file, 'r') as f:
            id_map = json.load(f)
        return index, id_map
    except FileNotFoundError:
        print("❌ 错误: 向量索引文件未找到，请先构建向量索引。")
        sys.exit(1)


def get_similar_images(image_path, index, id_map, similarity, num=5):
    """获取相似图片"""
    # 读取图片
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)

    # 检测服装物品
    clothing_detector = ClothingDetector()
    segmented_images = clothing_detector.detect_clothes(image_np)

    if not segmented_images:
        print("未在图像中检测到服装物品，使用整张图像进行相似度检测。")
        segmented_images = [image_np.copy()]

    # 提取特征向量
    feature_vectors = []
    for img in segmented_images:
        feature = similarity.extract_feature(img)
        feature_vectors.append(feature)

    # 计算平均特征向量
    avg_feature_vector = np.mean(feature_vectors, axis=0)

    # 搜索相似图片
    distances, indices = index.kneighbors([avg_feature_vector], n_neighbors=num)

    similar_images = []
    for idx, dist in zip(indices[0], distances[0]):
        image_id = id_map[idx]
        similarity_score = 1 - dist  # 根据距离计算相似度
        similar_images.append({
            "id": image_id,
            "similarity": similarity_score,
            "processed_image_base64": ""  # 这里暂时不提供处理后的图片，可根据需求添加
        })

    return similar_images


def save_similar_images_from_db(image_sub_dir, similar_images):
    """从数据库获取图片数据并保存到指定文件夹"""
    # 提取所有相似图片的ID
    ids = [item['id'] for item in similar_images]
    print(f"🔍 尝试批量查询的图片ID: {ids}")

    # 从数据库中获取图片数据
    image_data_dict = select_multiple_image_data_by_ids(ids)
    print(f"从数据库获取到的图片数据数量: {len(image_data_dict)}")

    for idx, item in enumerate(similar_images, start=1):
        image_id = item['id']
        # 查找对应的图片数据
        image_data = image_data_dict.get(image_id)
        if image_data:
            binary_image_data = image_data['splitted_image_data']
            try:
                # 将二进制数据转换为 PIL 图像
                img = Image.open(io.BytesIO(binary_image_data))
                new_image_name = f"similar_{idx:02d}.png"
                new_image_path = image_sub_dir / new_image_name
                img.save(new_image_path)
                print(f"✅ 相似图片 {new_image_path} 已保存")
            except Exception as e:
                print(f"❌ 保存图片时出错: {e}")
        else:
            print(f"❌ 未找到图片数据，图片ID: {image_id}")


def save_similar_images(image_sub_dir, similar_images):
    """保存相似图片并生成文本文件"""
    # 如果文件夹已存在，先清空
    if image_sub_dir.exists():
        for item in image_sub_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
    image_sub_dir.mkdir(parents=True, exist_ok=True)

    details = []
    for idx, item in enumerate(similar_images):
        try:
            details.append(f"名称: {item['id']}, 相似度: {item['similarity']:.4f}")
        except Exception as e:
            print(f"❌ 处理相似图片信息时出错: {e}")

    # 保存细节信息到文本文件
    details_file = image_sub_dir / "similarity_details.txt"
    with open(details_file, 'w') as f:
        for detail in details:
            f.write(detail + '\n')
    print(f"✅ 相似图片细节信息已保存到 {details_file}")


def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()

    # 找到项目根目录
    root_dir = find_project_root()
    print(f"📁 项目根目录: {root_dir}")

    # 确认图片目录路径
    image_dir = root_dir / "assets" / "test_image_similarity"
    if not image_dir.exists() or not image_dir.is_dir():
        print(f"❌ 错误: 图片目录不存在: {image_dir}")
        sys.exit(1)

    # 加载图片
    image_files = load_images(image_dir)

    # 创建data/test_similarity目录
    data_dir = root_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    test_dir = data_dir / "test_similarity"
    test_dir.mkdir(parents=True, exist_ok=True)

    # 初始化 ImageSimilarity 类
    # similarity = ImageSimilarity()
    similarity = ImageSimilarityViT()

    # 加载向量索引
    index, id_map = load_vector_index(root_dir)

    # 批量处理图片
    for image_path in image_files:
        image_name = Path(image_path).stem
        image_sub_dir = test_dir / f"test_{image_name}"

        # 获取相似图片
        similar_images = get_similar_images(image_path, index, id_map, similarity)
        print(f"🔍 为图片 {image_path} 找到的相似图片数量: {len(similar_images)}")

        # 保存相似图片并生成文本文件
        save_similar_images(image_sub_dir, similar_images)

        # 从数据库中获取并保存相似图片
        save_similar_images_from_db(image_sub_dir, similar_images)


if __name__ == "__main__":
    main()
