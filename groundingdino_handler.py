# groundingdino_handler.py
import sys
import os
from pathlib import Path
from PIL import Image
import numpy as np

# 导入刚才创建的模块
from image_processing import split_image_vertically, run_inference, prepare_transform

# 设置 Python 路径
current_dir = Path(__file__).parent.resolve()
groundingdino_path = current_dir / 'GroundingDINO'
sys.path.append(str(groundingdino_path))

# 导入必要的库
from GroundingDINO.groundingdino.util.inference import load_model

def detect_clothes_in_image(image, FRAME_WINDOW=None):
    """
    该函数用于在输入的图像中检测服装区域，并返回检测到的服装区域的边界框列表。

    函数首先会加载 GroundingDINO 模型所需的配置文件和权重文件，然后对输入图像进行预处理，
    包括将图像分割成多个片段，接着使用模型对每个片段进行推理，检测出其中的服装区域，
    最后将检测到的服装区域的边界框信息存储在列表中并返回。

    参数:
    image (numpy.ndarray): 输入的图像数据，以 numpy 数组的形式表示，通常是通过 PIL 图像转换而来。
    FRAME_WINDOW (streamlit.delta_generator.DeltaGenerator): Streamlit 中用于显示图像的窗口对象，
                                                           用于在检测过程中实时显示处理后的图像。

    返回:
    list: 一个包含检测到的服装区域边界框的列表，每个边界框以 numpy 数组的形式表示。
          如果检测过程中出现错误或未检测到任何服装区域，则返回一个空列表。

    抛出:
    FileNotFoundError: 如果指定的权重文件未找到，将抛出此异常，提示用户确保权重文件已放置在项目根目录中。
    Exception: 如果在图像分割、模型推理或其他处理步骤中出现错误，将捕获该异常并打印错误信息，
               同时返回一个空列表。
    """
    # Model Backbone
    CONFIG_PATH = groundingdino_path / 'groundingdino' / 'config' / 'GroundingDINO_SwinT_OGC.py'
    WEIGHTS_PATH = current_dir / 'groundingdino_swint_ogc.pth'

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

# # 示例调用
# if __name__ == "__main__":
#     image_path = "your_image_path.jpg"  # 替换为实际的图像路径
#     result_bboxes = detect_clothes_in_image(image_path)
#     for idx, bbox in enumerate(result_bboxes):
#         img = Image.fromarray(bbox)
#         img.show()