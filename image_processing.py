# image_processing.py
import numpy as np
from PIL import Image
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.util.inference import predict, annotate
import logging
from typing import List, Tuple

# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def split_image_vertically(image: np.ndarray, segment_height: int) -> List[np.ndarray]:
    """
    将图像纵向分割成多段，每段的高度为 segment_height。

    参数:
    - image (np.ndarray): 输入图像，形状为 (height, width, channels)。
    - segment_height (int): 每段图像的高度。

    返回:
    - List[np.ndarray]: 包含所有分割段的图像列表。

    异常:
    - ValueError: 如果 segment_height 大于图像高度，抛出异常。
    """
    height, width, _ = image.shape
    if segment_height > height:
        raise ValueError("Segment height cannot be larger than image height.")

    segments = []
    for i in range(0, height, segment_height):
        segment = image[i:i + segment_height, :, :]
        segments.append(segment)

    logger.info(f"Image split into {len(segments)} segments.")
    return segments


def combine_segments_vertically(segments: List[np.ndarray], original_height: int, original_width: int) -> np.ndarray:
    """
    将分割的图像段重新拼接成原始图像大小。

    参数:
    - segments (List[np.ndarray]): 分割图像的段列表。
    - original_height (int): 原始图像的高度。
    - original_width (int): 原始图像的宽度。

    返回:
    - np.ndarray: 拼接后的图像。
    """
    combined_image = np.vstack(segments)
    logger.info(f"Segments combined into one image of size ({original_height}, {original_width}).")
    return combined_image[:original_height, :original_width, :]


def pad_image(image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    """
    将图像填充或缩放为目标大小，并居中显示。

    参数:
    - image (Image.Image): 输入图像。
    - target_size (Tuple[int, int]): 目标图像的大小 (宽, 高)。

    返回:
    - Image.Image: 填充后的图像。
    """
    iw, ih = image.size  # 原始图像的尺寸
    w, h = target_size  # 目标图像的尺寸
    scale = min(w / iw, h / ih)  # 缩放比例
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.Resampling.LANCZOS)
    new_image = Image.new('RGB', target_size, (255, 255, 255))  # 创建白色背景
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))  # 居中粘贴
    logger.info(f"Image padded and resized to {target_size}.")
    return new_image


def run_inference(model, transform, segments: List[np.ndarray], TEXT_PROMPT: str, BOX_THRESHOLD: float, FRAME_WINDOW) -> List[np.ndarray]:
    """
    对每个图像段进行推理，检测目标并注释。

    参数:
    - model: 用于推理的模型。
    - transform: 图像转换函数。
    - segments (List[np.ndarray]): 输入图像段列表。
    - TEXT_PROMPT (str): 目标检测的文本提示。
    - BOX_THRESHOLD (float): 边界框置信度阈值。
    - FRAME_WINDOW: 用于展示结果的窗口对象。

    返回:
    - List[np.ndarray]: 包含所有目标检测图像段的列表。
    """
    bboxes = []  # 存储每个 bounding box 的图像区域
    annotated_segments = []

    for segment in segments:
        try:
            segment_transformed, _ = transform(Image.fromarray(segment), None)
            boxes, logits, phrases = predict(
                model=model,
                device="cpu",
                image=segment_transformed,
                caption=TEXT_PROMPT,
                box_threshold=BOX_THRESHOLD,
                text_threshold=0.25
            )
            annotated_segment, detection = annotate(segment, boxes=boxes, logits=logits, phrases=phrases)
            annotated_segments.append(annotated_segment)

            # 提取每个 bounding box 的图像区域并存储在 bboxes 列表中
            for box in detection.xyxy:
                x1, y1, x2, y2 = map(int, box)
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                # 过滤掉宽度小于1/5图像宽度的bbox
                if max(160, segment.shape[1] / 5) <= bbox_width <= bbox_height:
                    bbox_image = segment[y1:y2, x1:x2]
                    bboxes.append(bbox_image)
        except Exception as e:
            logger.error(f"Error during inference: {e}")

    # 拼接所有注释过的段以形成完整图像
    if FRAME_WINDOW is not None:
        annotated_image = combine_segments_vertically(annotated_segments, segments[0].shape[0] * len(segments),
                                                      segments[0].shape[1])
        FRAME_WINDOW.image(annotated_image, channels='BGR')

    logger.info(f"Inference completed. {len(bboxes)} bounding boxes detected.")
    return bboxes


def prepare_transform() -> T.Compose:
    """
    准备用于图像转换的预处理函数。

    返回:
    - T.Compose: 由多个图像处理步骤组成的转换对象。
    """
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    logger.info("Transform preparation completed.")
    return transform
