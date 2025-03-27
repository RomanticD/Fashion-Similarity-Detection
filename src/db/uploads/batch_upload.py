import sys
import os
from pathlib import Path
from PIL import Image
import numpy as np

# 设置 Python 路径
root_dir = Path(__file__).parent.resolve()
while not (root_dir / "setup.py").exists() and not (root_dir / ".git").exists():
    root_dir = root_dir.parent

groundingdino_path = root_dir / 'GroundingDINO'
sys.path.append(str(groundingdino_path))

# 导入相关模块
from src.core.groundingdino_handler import detect_clothes_in_image
from src.core.image_upload import process_and_upload_image


def process_single_image(image_path):
    """处理单个图像并上传分割后的部分"""
    print(f"正在处理图像: {image_path}")

    # 获取图像名称（不含扩展名）
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # 检查图像文件是否存在
    if not image_path.exists():
        print(f"图像文件 '{image_path}' 不存在，跳过处理。")
        return []

    try:
        # 打开图像并转换为 numpy 数组
        image = Image.open(image_path)
        image_np = np.array(image)

        # 调用检测函数获取分割后的图像数组
        result_bboxes = detect_clothes_in_image(image_np)

        # 创建数据目录
        data_dir = root_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # 创建原始图像对应的子目录
        image_dir = data_dir / image_name
        image_dir.mkdir(parents=True, exist_ok=True)

        # 处理并上传每个分割图像
        uploaded_paths = []
        for idx, img_array in enumerate(result_bboxes):
            filename = f"segment_{idx}.png"
            save_path = image_dir / filename  # 保存到对应的子目录

            # 处理并上传图像
            processed_path = process_and_upload_image(img_array, idx, save_path, image_name)
            if processed_path:
                uploaded_paths.append(processed_path)
                print(f"已上传分割图片: {processed_path}")

        return uploaded_paths

    except Exception as e:
        print(f"处理图像 '{image_path}' 时出错: {e}")
        return []


def batch_process_screenshots():
    """批量处理 ScreenShots 目录中的所有图像"""
    # 获取根目录
    root_dir = Path(__file__).parent.resolve()
    while not (root_dir / "setup.py").exists() and not (root_dir / ".git").exists():
        root_dir = root_dir.parent

    # 检查是否存在 ScreenShots 文件夹
    screenshots_dir = root_dir / "ScreenShots"
    if not screenshots_dir.exists() or not screenshots_dir.is_dir():
        raise FileNotFoundError(f"'{screenshots_dir}' 文件夹不存在，请确保下载资源文件并将该文件夹存在于项目根目录中。")

    # 获取所有图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    image_files = []

    for ext in image_extensions:
        image_files.extend(list(screenshots_dir.glob(f'*{ext}')))
        image_files.extend(list(screenshots_dir.glob(f'*{ext.upper()}')))  # 同时检查大写扩展名

    if not image_files:
        print(f"在 '{screenshots_dir}' 目录中未找到任何图像文件。")
        return

    print(f"找到 {len(image_files)} 个图像文件待处理。")

    # 处理所有图像
    all_uploaded_paths = []
    for image_path in image_files:
        uploaded_paths = process_single_image(image_path)
        all_uploaded_paths.extend(uploaded_paths)

    print(f"\n批量处理完成！共上传了 {len(all_uploaded_paths)} 个分割图像。")
    return all_uploaded_paths


if __name__ == "__main__":
    try:
        # 执行批量处理
        uploaded_files = batch_process_screenshots()

        # 打印处理结果
        if uploaded_files:
            print("\n成功上传的文件:")
            for path in uploaded_files:
                print(f" - {path}")
        else:
            print("\n没有成功上传任何文件。")

    except Exception as e:
        print(f"批量处理过程中出错: {e}")