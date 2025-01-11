# import sys
# import os
# from pathlib import Path
# import numpy as np
# from PIL import Image
# from GroundingDINO.groundingdino.util.inference import load_model
# from image_processing import split_image_vertically, prepare_transform, run_inference
#
# # 设置路径
# current_dir = Path(__file__).parent.resolve()
# groundingdino_path = current_dir / 'GroundingDINO'
# sys.path.append(str(groundingdino_path))
#
# # 配置模型路径
# CONFIG_PATH = groundingdino_path / 'groundingdino' / 'config' / 'GroundingDINO_SwinT_OGC.py'
# WEIGHTS_PATH = current_dir / 'groundingdino_swint_ogc.pth'
#
#
# # 加载模型
# def load_groundingdino_model():
#     if not WEIGHTS_PATH.exists():
#         raise FileNotFoundError(f"权重文件未找到，请确保将 '{WEIGHTS_PATH}' 放置在项目根目录中。")
#     try:
#         return load_model(str(CONFIG_PATH), str(WEIGHTS_PATH), device='cpu')
#     except Exception as e:
#         raise RuntimeError(f"加载模型时出错: {e}")
#
#
# # 获取变换函数
# def get_transform():
#     try:
#         return prepare_transform()
#     except Exception as e:
#         raise RuntimeError(f"获取图像变换函数时出错: {e}")
#
#
# # 处理图像分段
# def process_image(image, transform):
#     try:
#         image_transformed, _ = transform(image, None)
#         segment_height = image.shape[1] * 3
#         segments = split_image_vertically(image, segment_height)
#         return segments
#     except Exception as e:
#         raise RuntimeError(f"处理图像时出错: {e}")
#
#
# # 推理并返回检测框
# def detect_clothes(model, transform, segments, TEXT_PROMPT, BOX_THRESHOLD):
#     try:
#         return run_inference(model, transform, segments, TEXT_PROMPT, BOX_THRESHOLD)
#     except Exception as e:
#         raise RuntimeError(f"服装检测出错: {e}")
