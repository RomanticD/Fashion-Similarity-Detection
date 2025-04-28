# src/core/image_processing.py
import numpy as np
from PIL import Image
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.util.inference import predict, annotate
import logging
from typing import List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    A class providing image processing utilities for the detection pipeline.
    """

    MIN_PAD_WIDTH = 448
    MIN_PAD_HEIGHT = 1000

    MIN_WIDTH = 160

    def split_image_vertically(self, image: np.ndarray, segment_height: int) -> List[np.ndarray]:
        """
        Split an image vertically into multiple segments.

        Args:
            image (np.ndarray): Input image with shape (height, width, channels).
            segment_height (int): Height of each segmented image.

        Returns:
            List[np.ndarray]: A list of segmented images.

        Raises:
            ValueError: If segment_height is greater than image height.
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

    def split_image_horizontally(self, image: np.ndarray, segment_width: int) -> List[np.ndarray]:
        """
        Split an image horizontally into multiple segments.

        Args:
            image (np.ndarray): Input image with shape (height, width, channels).
            segment_width (int): Width of each segmented image.

        Returns:
            List[np.ndarray]: A list of segmented images.

        Raises:
            ValueError: If segment_width is greater than image width.
        """
        height, width, _ = image.shape
        if segment_width > width:
            raise ValueError("Segment width cannot be larger than image width.")

        segments = []
        for i in range(0, width, segment_width):
            segment = image[:, i:i + segment_width, :]
            segments.append(segment)

        logger.info(f"Image split into {len(segments)} segments.")
        return segments

    def combine_segments_vertically(self, segments: List[np.ndarray], original_height: int,
                                    original_width: int) -> np.ndarray:
        """
        Recombine segmented images into an image of the original size.

        Args:
            segments (List[np.ndarray]): A list of segmented image segments.
            original_height (int): The height of the original image.
            original_width (int): The width of the original image.

        Returns:
            np.ndarray: The combined image.
        """
        combined_image = np.vstack(segments)
        logger.info(f"Segments combined into one image of size ({original_height}, {original_width}).")
        return combined_image[:original_height, :original_width, :]

    def combine_segments_horizontally(self, segments: List[np.ndarray], original_height: int,
                                      original_width: int) -> np.ndarray:
        """
        Recombine horizontally segmented images into an image of the original size.

        Args:
            segments (List[np.ndarray]): A list of segmented image segments.
            original_height (int): The height of the original image.
            original_width (int): The width of the original image.

        Returns:
            np.ndarray: The combined image.
        """
        combined_image = np.hstack(segments)
        logger.info(f"Segments combined horizontally into one image of size ({original_height}, {original_width}).")
        return combined_image[:, :original_width, :]

    def pad_image(self, image: Image.Image) -> Image.Image:
        """
        独立填充宽度和高度至最小尺寸，不足时居中填充，不缩放原图
        """
        iw, ih = image.size
        target_width = max(iw, self.MIN_PAD_WIDTH)  # 宽度至少448
        target_height = max(ih, self.MIN_PAD_HEIGHT)  # 高度至少1000

        # 计算各方向填充量
        pad_w = target_width - iw
        pad_h = target_height - ih
        left_pad = pad_w // 2
        right_pad = pad_w - left_pad
        top_pad = pad_h // 2
        bottom_pad = pad_h - top_pad

        # 创建填充后的图像（居中粘贴原图）
        new_image = Image.new('RGB', (target_width, target_height), (128, 128, 128))  # 灰色背景
        new_image.paste(image, (left_pad, top_pad))  # 居中粘贴

        logger.info(f"Image padded to {target_width}x{target_height} (original: {iw}x{ih})")
        return new_image

    def run_inference(self, model, transform, image: np.ndarray, text_prompt: str,
                      box_threshold: float, frame_window=None) -> List[np.ndarray]:
        """
        Run inference on the image, detect targets, and annotate them.

        Args:
            model: The model for inference.
            transform: The image transformation function.
            image (np.ndarray): The input image.
            text_prompt (str): The text prompt for target detection.
            box_threshold (float): The confidence threshold for bounding boxes.
            frame_window: The window to display results.

        Returns:
            List[np.ndarray]: A list of detected image segments.
        """
        original_pil = Image.fromarray(image)
        orig_width, orig_height = original_pil.size
        # 使用类常量判断是否需要 padding
        is_padded = orig_width < self.MIN_PAD_WIDTH or orig_height < self.MIN_PAD_HEIGHT

        if is_padded:
            logger.info(f"原始图像尺寸不足 ({orig_width}x{orig_height})，需要进行填充")
            padded_pil = self.pad_image(original_pil)
            image = np.array(padded_pil)
        else:
            logger.info(f"原始图像尺寸足够 ({orig_width}x{orig_height})，无需填充")

        height, width, _ = image.shape
        aspect_ratio = width / height
        logger.info(f"图像宽高比: {aspect_ratio:.2f}, 尺寸: {width}x{height}")

        if aspect_ratio <= 1 / 3:  # 1:3 or taller
            segment_height = width * 3
            logger.info(f"图像过高，需要垂直分割，分段高度: {segment_height}")
            segments = self.split_image_vertically(image, segment_height)
        elif aspect_ratio >= 3:  # 3:1 or wider
            segment_width = height * 3
            logger.info(f"图像过宽，需要水平分割，分段宽度: {segment_width}")
            segments = self.split_image_horizontally(image, segment_width)
        else:  # Balanced aspect ratio
            logger.info("图像宽高比适中，无需分割")
            segments = [image]

        bboxes = []
        annotated_segments = []
        total_boxes_detected = 0

        for i, segment in enumerate(segments):
            logger.info(f"处理图像分段 {i+1}/{len(segments)}, 尺寸: {segment.shape[1]}x{segment.shape[0]}")
            try:
                segment_transformed, _ = transform(Image.fromarray(segment), None)
                boxes, logits, phrases = predict(
                    model=model,
                    device="cpu",
                    image=segment_transformed,
                    caption=text_prompt,
                    box_threshold=box_threshold,
                    text_threshold=0.25
                )
                
                num_boxes = len(boxes)
                total_boxes_detected += num_boxes
                logger.info(f"在分段 {i+1} 中检测到 {num_boxes} 个潜在边界框")
                
                annotated_segment, detection = annotate(segment, boxes=boxes, logits=logits, phrases=phrases)
                annotated_segments.append(annotated_segment)

                if num_boxes > 0:
                    width_min = max(160, segment.shape[1] / 5)  # 最小宽度要求
                    logger.info(f"当前分段的最小宽度要求: {width_min:.1f}像素")
                
                for j, box in enumerate(detection.xyxy):
                    x1, y1, x2, y2 = map(int, box)
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    
                    logger.info(f"边界框 #{j+1}: 坐标=({x1},{y1},{x2},{y2}), 尺寸={bbox_width}x{bbox_height}")
                    
                    width_min = max(160, segment.shape[1] / 5)  # 保留原有逻辑

                    if is_padded:
                        # 已填充图像的处理逻辑
                        if bbox_width >= width_min and bbox_height >= self.MIN_WIDTH:
                            logger.info(f"接受边界框 #{j+1}: 宽度={bbox_width}≥{width_min}, 高度={bbox_height}≥{self.MIN_WIDTH}")
                            bbox_image = segment[y1:y2, x1:x2]
                            bboxes.append(bbox_image)
                        else:
                            if bbox_width < width_min:
                                logger.warning(f"跳过边界框 #{j+1}: 宽度不足 ({bbox_width}<{width_min})")
                            if bbox_height < self.MIN_WIDTH:
                                logger.warning(f"跳过边界框 #{j+1}: 高度不足 ({bbox_height}<{self.MIN_WIDTH})")
                    else:
                        # 未填充图像的处理逻辑
                        if width_min <= bbox_width <= bbox_height:
                            logger.info(f"接受边界框 #{j+1}: {width_min}≤宽度({bbox_width})≤高度({bbox_height})")
                            bbox_image = segment[y1:y2, x1:x2]
                            bboxes.append(bbox_image)
                        else:
                            if bbox_width < width_min:
                                logger.warning(f"跳过边界框 #{j+1}: 宽度不足 ({bbox_width}<{width_min})")
                            if bbox_width > bbox_height:
                                logger.warning(f"跳过边界框 #{j+1}: 宽度({bbox_width})>高度({bbox_height})")
            except Exception as e:
                logger.error(f"推理过程中出错: {e}")
                import traceback
                logger.debug(traceback.format_exc())

        if frame_window is not None:
            if aspect_ratio <= 1 / 3:
                annotated_image = self.combine_segments_vertically(
                    annotated_segments,
                    segments[0].shape[0] * len(segments),
                    segments[0].shape[1]
                )
            elif aspect_ratio >= 3:
                annotated_image = self.combine_segments_horizontally(
                    annotated_segments,
                    segments[0].shape[0],
                    segments[0].shape[1] * len(segments)
                )
            else:
                annotated_image = segments[0]
            frame_window.image(annotated_image, channels='BGR')

        logger.info(f"推理完成。检测到 {total_boxes_detected} 个边界框，保留 {len(bboxes)} 个符合条件的区域。")
        return bboxes

    def prepare_transform(self) -> T.Compose:
        """
        Prepare the preprocessing function for image transformation.

        Returns:
            T.Compose: A transformation object.
        """
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        logger.info("Transform preparation completed.")
        return transform
