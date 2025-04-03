#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试脚本: 直接运行切图算法，处理图片并保存分割结果
用法: python test_upload.py [--image_dir Assets/] [--force]
"""

import argparse
import base64
import sys
from pathlib import Path
import shutil
from PIL import Image
import io
import numpy as np

# 导入必要的模块
from src.core.groundingdino_handler import ClothingDetector
from src.utils.data_conversion import base64_to_numpy


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='测试图片上传API')
    parser.add_argument('--image_dir', type=str, default='Assets/',
                        help='要上传的图片所在目录 (默认: Assets/)')
    parser.add_argument('--force', action='store_true', default=True,
                        help='即使未检测到服装也强制处理 (默认: True)')
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


def image_to_base64(image_path):
    """将图片文件转换为Base64编码字符串"""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    except Exception as e:
        print(f"❌ 错误: 读取图片文件时出错: {e}")
        sys.exit(1)


def base64_to_image(base64_str):
    """将Base64编码字符串转换为PIL图像"""
    img_data = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_data))
    return img


def save_segmented_images(image_sub_dir, segmented_images):
    """保存分割后的图片到对应的test_X文件夹中"""
    image_sub_dir.mkdir(parents=True, exist_ok=True)
    for idx, img_array in enumerate(segmented_images):
        try:
            filename = f"segment_{idx}.png"
            save_path = image_sub_dir / filename
            img = Image.fromarray(img_array)
            img.save(save_path)
            print(f"✅ 分割图片 {save_path} 已保存")
        except Exception as e:
            print(f"❌ 保存分割图片 {save_path} 时出错: {e}")


def process_image(image_path, test_dir, force_process=True):
    """处理单张图片并保存分割结果"""
    print(f"🔍 准备处理图片: {image_path}")

    # 读取图片并转换为numpy数组
    base64_image = image_to_base64(image_path)
    clean_base64 = base64_image.split(',', 1)[1] if base64_image.startswith('data:') else base64_image
    image_np = base64_to_numpy(clean_base64)

    # 确保图像是RGB格式
    def ensure_rgb_format(image_np):
        if len(image_np.shape) == 3 and image_np.shape[2] == 4:
            img = Image.fromarray(image_np)
            rgb_img = img.convert('RGB')
            return np.array(rgb_img)
        elif len(image_np.shape) == 3 and image_np.shape[2] == 3:
            return image_np
        elif len(image_np.shape) == 2:
            img = Image.fromarray(image_np)
            rgb_img = img.convert('RGB')
            return np.array(rgb_img)
        else:
            img = Image.fromarray(image_np)
            rgb_img = img.convert('RGB')
            return np.array(rgb_img)

    image_np = ensure_rgb_format(image_np)

    # 初始化检测器
    clothing_detector = ClothingDetector()
    clothing_detector.box_threshold = 0.15

    # 检测服装物品
    try:
        segmented_images = clothing_detector.detect_clothes(image_np)
        if not segmented_images and not force_process:
            print("未在图像中检测到服装物品，跳过处理")
            return
        if not segmented_images and force_process:
            print("未检测到服装，但因强制处理标志而继续处理整张图像")
            segmented_images = [image_np.copy()]
    except Exception as e:
        print(f"服装检测错误: {e}")
        return

    # 保存分割后的图片
    image_name = Path(image_path).stem
    image_sub_dir = test_dir / f"test_{image_name}"
    save_segmented_images(image_sub_dir, segmented_images)


def clean_unused_folders(test_dir, image_names):
    """删除test文件夹下与当前图片名称不同的文件夹"""
    if test_dir.exists():
        for item in test_dir.iterdir():
            if item.is_dir():
                expected_prefix = "test_"
                if item.name.startswith(expected_prefix):
                    folder_image_name = item.name[len(expected_prefix):]
                    if folder_image_name not in image_names:
                        shutil.rmtree(item)
                        print(f"已删除文件夹: {item}")


def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()

    # 找到项目根目录
    root_dir = find_project_root()
    print(f"📁 项目根目录: {root_dir}")

    # 确认图片目录路径
    image_dir = root_dir / args.image_dir
    if not image_dir.exists() or not image_dir.is_dir():
        print(f"❌ 错误: 图片目录不存在: {image_dir}")
        sys.exit(1)

    # 获取目录下的所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    image_files = []
    image_names = []
    for ext in image_extensions:
        image_files.extend(list(image_dir.glob(f'*{ext}')))
        image_files.extend(list(image_dir.glob(f'*{ext.upper()}')))  # 检查大写扩展名

    if not image_files:
        print(f"❌ 错误: 图片目录中没有找到图片文件: {image_dir}")
        sys.exit(1)

    print(f"✅ 找到 {len(image_files)} 个图片文件")

    # 创建data/test_groundingDINO目录
    data_dir = root_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    test_dir = data_dir / "test_groundingDINO"
    test_dir.mkdir(parents=True, exist_ok=True)

    # 批量处理图片
    for image_path in image_files:
        image_name = Path(image_path).stem
        image_names.append(image_name)
        image_sub_dir = test_dir / f"test_{image_name}"
        if image_sub_dir.exists():
            print(f"文件夹 {image_sub_dir} 已存在，跳过切图操作")
            continue
        process_image(image_path, test_dir, args.force)

    # 删除无用的文件夹
    clean_unused_folders(test_dir, image_names)


if __name__ == "__main__":
    main()
