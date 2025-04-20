#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path
import base64
import time

# 导入 ImageSimilarity 类、ImageSimilarityViT 类和 ImageSimilarityDINOv2 类
from src.core.image_similarity import ImageSimilarity
from src.core.image_similarity_vit import ImageSimilarityViT
from src.core.image_similarity_DINOv2 import ImageSimilarityDINOv2


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='测试不同模型的图片相似度检测性能')
    parser.add_argument('--metric', type=str, default='cosine', choices=['cosine', 'euclidean', 'manhattan'],
                        help='选择相似度计算方法 (cosine, euclidean, manhattan)')
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

    if len(image_files) != 2:
        print(f"❌ 错误: 图片目录中应包含两张图片: {image_dir}")
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


def calculate_similarity(model, image_path1, image_path2, metric='cosine'):
    """计算两张图片的相似度"""
    start_time = time.time()
    feature1 = model.extract_feature(image_path1)
    feature2 = model.extract_feature(image_path2)
    single_dict = {'image1': feature1}
    images_dict = {'image2': feature2}
    similarity_result = model.compare_similarities(single_dict, images_dict, metric=metric)
    similarity = similarity_result[0][1]
    extraction_time = time.time() - start_time
    return similarity, extraction_time


def save_results(image_sub_dir, model_name, similarity, extraction_time, metric, file):
    """保存测试结果到文本文件"""
    file.write(f"模型: {model_name}\n")
    file.write(f"相似度计算方法: {metric}\n")
    file.write(f"相似度: {similarity:.4f}\n")
    file.write(f"特征提取时间: {extraction_time:.4f}s\n")


def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()
    metric = args.metric

    # 找到项目根目录
    root_dir = find_project_root()
    print(f"📁 项目根目录: {root_dir}")

    # 确认图片目录路径
    image_dir = root_dir / "assets" / "test_model"
    if not image_dir.exists() or not image_dir.is_dir():
        print(f"❌ 错误: 图片目录不存在: {image_dir}")
        sys.exit(1)

    # 创建data/test_model目录
    data_dir = root_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    test_dir = data_dir / "test_model"
    test_dir.mkdir(parents=True, exist_ok=True)

    # 初始化模型
    resnet_model = ImageSimilarity()
    vit_model = ImageSimilarityViT()
    dinov2_model = ImageSimilarityDINOv2()

    # 遍历每个测试组
    for group_dir in image_dir.iterdir():
        if group_dir.is_dir():
            group_name = group_dir.name
            image_sub_dir = test_dir / group_name
            image_sub_dir.mkdir(parents=True, exist_ok=True)

            # 加载图片
            image_files = load_images(group_dir)
            image_path1, image_path2 = image_files

            # 打开文件以保存结果
            details_file = image_sub_dir / f"{metric}_similarity_details.txt"
            with open(details_file, 'w') as f:
                # 计算 ResNet50 模型的相似度，固定使用余弦相似度
                start_time = time.time()
                feature1 = resnet_model.extract_feature(image_path1)
                feature2 = resnet_model.extract_feature(image_path2)
                resnet_similarity = resnet_model.cosine_similarity(feature1, feature2)
                resnet_time = time.time() - start_time
                save_results(image_sub_dir, "ResNet50", resnet_similarity, resnet_time, 'cosine', f)

                # 不同模型结果间空一行
                f.write("\n")

                # 计算 ViT 模型的相似度，使用指定的相似度计算方法
                vit_similarity, vit_time = calculate_similarity(vit_model, image_path1, image_path2, metric=metric)
                save_results(image_sub_dir, "ViT", vit_similarity, vit_time, metric, f)

                # 不同模型结果间空一行
                f.write("\n")

                # 计算 DINOv2 模型的相似度，使用指定的相似度计算方法
                dinov2_similarity, dinov2_time = calculate_similarity(dinov2_model, image_path1, image_path2, metric=metric)
                save_results(image_sub_dir, "DINOv2", dinov2_similarity, dinov2_time, metric, f)

            print(f"✅ 相似图片细节信息已保存到 {details_file}")


if __name__ == "__main__":
    main()
