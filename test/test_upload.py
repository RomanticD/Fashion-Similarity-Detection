#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试脚本: 模拟上传图片到新的API端点
用法: python test_upload.py [--host http://localhost:5001] [--image Assets/IMAGE_NAME.png] [--force]
"""

import argparse
import base64
import json
import os
import sys
import time
import requests
from pathlib import Path


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='测试图片上传API')
    parser.add_argument('--host', type=str, default='http://localhost:5001',
                        help='API服务器地址 (默认: http://localhost:5001)')
    parser.add_argument('--image', type=str, default='Assets/test.PNG',
                        help='要上传的图片路径 (默认: Assets/test.PNG)')
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


def upload_image(host_url, image_path, force_process=True):
    """上传图片到API端点并显示结果"""
    print(f"🔍 准备上传图片: {image_path}")

    # 转换图片为Base64
    base64_image = image_to_base64(image_path)
    print(f"✅ 图片已转换为Base64 (显示前20个字符): {base64_image[:20]}...")

    # 准备请求数据
    image_name = Path(image_path).stem  # 获取文件名（不含扩展名）
    payload = {
        "image_base64": base64_image,
        "image_name": f"test_{image_name}",
        "force_process": force_process
    }

    # 打印请求详情
    print(f"📤 发送请求到: {host_url}/upload_image")
    print(
        f"📋 请求数据: {{'image_base64': '(base64数据省略)', 'image_name': '{payload['image_name']}', 'force_process': {force_process}}}")

    # 发送请求
    start_time = time.time()
    try:
        response = requests.post(
            f"{host_url}/upload_image",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120  # 设置较长的超时时间，因为处理可能需要一段时间
        )
        elapsed_time = time.time() - start_time

        # 确保响应是JSON格式
        try:
            result = response.json()
        except json.JSONDecodeError:
            print(f"❌ 错误: 服务器返回的不是有效的JSON数据: {response.text[:100]}")
            return

        # 打印响应
        print(f"\n⏱️ 请求耗时: {elapsed_time:.2f} 秒")
        print(f"📥 响应状态码: {response.status_code}")

        # 格式化输出JSON结果
        formatted_json = json.dumps(result, ensure_ascii=False, indent=2)
        print(f"📄 响应内容:\n{formatted_json}")

        # 如果成功，显示更多信息
        if result.get("success"):
            request_id = result.get("request_id")
            segments = result.get("data", {}).get("segments", [])
            print(f"\n✅ 上传成功! 请求ID: {request_id}")
            print(f"🖼️ 检测到 {len(segments)} 个服装分割")

            # 显示每个分割的信息
            if segments:
                print("\n分割详情:")
                for i, segment in enumerate(segments, 1):
                    print(f"  {i}. 分割ID: {segment.get('splitted_image_id')}")
                    print(f"     路径: {segment.get('splitted_image_path')}")
        else:
            print(f"\n❌ 上传失败: {result.get('message')}")

    except requests.exceptions.RequestException as e:
        print(f"❌ 请求错误: {e}")


def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()

    # 找到项目根目录
    root_dir = find_project_root()
    print(f"📁 项目根目录: {root_dir}")

    # 确认完整的图片路径
    image_path = root_dir / args.image
    if not image_path.exists():
        print(f"❌ 错误: 图片文件不存在: {image_path}")
        sys.exit(1)

    print(f"✅ 找到图片文件: {image_path}")

    # 上传图片
    upload_image(args.host, image_path, args.force)


if __name__ == "__main__":
    main()