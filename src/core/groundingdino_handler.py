# groundingdino_handler.py
import sys
import os
from pathlib import Path
from PIL import Image
import numpy as np

# 导入刚才创建的模块
from image_processing import split_image_vertically, run_inference, prepare_transform

# 设置 Python 路径
root_dir = Path(__file__).parent.resolve()
while not (root_dir / "setup.py").exists() and not (root_dir / ".git").exists():
    root_dir = root_dir.parent

groundingdino_path = root_dir / 'GroundingDINO'
sys.path.append(str(groundingdino_path))

# 导入必要的库
from GroundingDINO.groundingdino.util.inference import load_model

"""
Detects clothing regions in the input image and returns bounding boxes of detected clothes.

This function loads the GroundingDINO model, processes the input image by splitting it into segments,
runs inference on each segment to detect clothing, and returns the bounding boxes of the detected clothes.

Parameters:
image (np.ndarray): Input image as a numpy array, typically obtained by converting a PIL image.
FRAME_WINDOW (optional): Streamlit window object for displaying the processed image during detection.

Returns:
list: A list of bounding boxes (as numpy arrays) for the detected clothing regions. 
      Returns an empty list if no clothes are detected or an error occurs.

Raises:
FileNotFoundError: If the weights file is not found.
Exception: If any error occurs during image processing or inference.
"""
def detect_clothes_in_image(image, FRAME_WINDOW=None):
    # Model Backbone
    CONFIG_PATH = groundingdino_path / 'groundingdino' / 'config' / 'GroundingDINO_SwinT_OGC.py'
    WEIGHTS_PATH = root_dir / 'groundingdino_swint_ogc.pth'

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

    # 检查图像文件是否存在
    if not image_path.exists():
        raise FileNotFoundError(f"图像文件 '{image_path}' 不存在，请确保文件路径正确。")

    # 打开图像并转换为 numpy 数组
    image = Image.open(image_path)

    # 转换为 numpy.ndarray
    image_np = np.array(image)

    # 调用检测函数进行服装检测
    result_bboxes = detect_clothes_in_image(image_np)

    # 显示检测结果
    for idx, bbox in enumerate(result_bboxes):
        img = Image.fromarray(bbox)
        img.show()