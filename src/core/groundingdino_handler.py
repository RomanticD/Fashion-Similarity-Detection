# src/core/groundingdino_handler.py
import sys
import os
from pathlib import Path
from PIL import Image
import numpy as np
from torch.utils.checkpoint import checkpoint

from src.core.image_similarity import load_images_from_arrays, extract_feature, compare_similarities, \
    load_single_image_feature_vector
from src.core.image_processing import split_image_vertically, run_inference, prepare_transform
from src.core.image_upload import process_and_upload_image

# 设置 Python 路径
root_dir = Path(__file__).parent.resolve()
while not (root_dir / "setup.py").exists() and not (root_dir / ".git").exists():
    root_dir = root_dir.parent

groundingdino_path = root_dir / 'GroundingDINO'
sys.path.append(str(groundingdino_path))

# 导入必要的库
from GroundingDINO.groundingdino.util.inference import load_model


def detect_clothes_in_image(image, FRAME_WINDOW=None):
    # Model Backbone
    CONFIG_PATH = groundingdino_path / 'groundingdino' / 'config' / 'GroundingDINO_SwinT_OGC.py'
    WEIGHTS_PATH = root_dir / 'src' / 'checkpoints' / 'groundingdino_swint_ogc.pth'

    # 检查权重文件是否存在
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"权重文件未找到，请确保将 '{WEIGHTS_PATH}' 放置在项目根目录中。")

    # 加载模型
    model = load_model(str(CONFIG_PATH), str(WEIGHTS_PATH), device='cpu')

    # 获取变换函数
    transform = prepare_transform()

    # 配置参数
    BOX_THRESHOLD = 0.3
    TEXT_PROMPT = "clothes"

    try:
        # 设定分段长度为图像宽度的3倍
        segment_height = image.shape[1] * 3
        segments = split_image_vertically(image, segment_height)

        # 进行服装检测
        clothes_bboxes = run_inference(model, transform, segments, TEXT_PROMPT, BOX_THRESHOLD, FRAME_WINDOW)

        return clothes_bboxes
    except Exception as e:
        print(f"检测过程中出错2: {e}")
        return []


def clear_directory(data_dir):
    # 确保路径是一个目录
    if data_dir.exists() and data_dir.is_dir():
        # 遍历目录中的所有文件并删除
        for file in data_dir.rglob('*'):  # rglob 匹配所有文件和文件夹
            if file.is_file():  # 仅删除文件，保持子文件夹
                file.unlink()  # 删除文件
            elif file.is_dir():  # 删除空的子文件夹
                file.rmdir()  # 删除空文件夹
    else:
        print(f"{data_dir} 不是一个有效的目录。")


# 示例调用
if __name__ == "__main__":
    # 根目录
    root_dir = Path(__file__).parent.resolve()

    while not (root_dir / "setup.py").exists() and not (root_dir / ".git").exists():
        root_dir = root_dir.parent

    # 检查是否存在 ScreenShots 文件夹
    screenshots_dir = root_dir / "ScreenShots"
    if not screenshots_dir.exists() or not screenshots_dir.is_dir():
        raise FileNotFoundError(f"'{screenshots_dir}' 文件夹不存在，请确保下载资源文件并将该文件夹存在于项目根目录中。")

    # 图像路径
    image_path = screenshots_dir / "01.jpeg"

    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # 检查图像文件是否存在
    if not image_path.exists():
        raise FileNotFoundError(f"图像文件 '{image_path}' 不存在，请确保文件路径正确。")

    # 打开图像并转换为 numpy 数组
    image = Image.open(image_path)
    image_np = np.array(image)

    # 调用检测函数获取分割后的图像数组
    result_bboxes = detect_clothes_in_image(image_np)

    # --------------------------
    # 创建数据目录并处理图像
    # --------------------------
    data_dir = root_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)  # 自动创建目录

    # 创建原始图像对应的子目录
    image_dir = data_dir / image_name
    image_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for idx, img_array in enumerate(result_bboxes):
        filename = f"segment_{idx}.png"
        save_path = image_dir / filename  # 保存到对应的子目录

        # 处理并上传图像 - 调用独立的函数
        processed_path = process_and_upload_image(img_array, idx, save_path, image_name)
        saved_paths.append(processed_path)
        print(f"已保存分割图片: {save_path}")
    # 加载分割图像特征（从保存的文件加载）
    segmented_features = {}
    for path in saved_paths:
        feature_dict = load_single_image_feature_vector(path)
        segmented_features.update(feature_dict)

    # 加载对比图片特征
    single_img_path = root_dir / "Assets" / "spilt_image_similarity_test_2.png"
    single_feature = load_single_image_feature_vector(single_img_path)

    # 执行相似度对比
    similarity_results = compare_similarities(single_feature, segmented_features)

    # 打印对比结果（保持原输出格式）
    print(f"\n对比图片: {single_img_path.name}")
    for img_name, similarity in sorted(similarity_results, key=lambda x: x[1], reverse=True):
        print(f"分割区域 {img_name}: 相似度 {similarity:.4f}")