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
        Pad or resize an image to the target size and center it.
        target_size: 448*896.

        Args:
            image (Image.Image): The input image.

        Returns:
            Image.Image: The padded image.
        """
        min_width, min_height = 448, 896
        iw, ih = image.size

        # Calculate padding for width and height independently
        pad_width = max(0, min_width - iw)
        pad_height = max(0, min_height - ih)
        final_w = iw + pad_width
        final_h = ih + pad_height

        # Create grey background (RGB 128,128,128) and paste original image in center
        padded_image = Image.new('RGB', (final_w, final_h), (128, 128, 128))
        padded_image.paste(image, (pad_width // 2, pad_height // 2))

        logger.info(f"Padded image from {iw}x{ih} to {final_w}x{final_h}")
        return padded_image

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
        is_padded = orig_width < 448 or orig_height < 896  # 新增：判断是否进行过padding

        # 仅当宽度<448 **或** 高度<896时执行填充
        if is_padded:
            logger.info(f"原图尺寸不足 ({orig_width}x{orig_height}), 执行填充处理")
            padded_pil = self.pad_image(original_pil)
            image = np.array(padded_pil)  # 转换回numpy数组
        else:
            logger.info(f"原图尺寸足够 ({orig_width}x{orig_height}), 无需填充")

        height, width, _ = image.shape
        aspect_ratio = width / height

        if aspect_ratio <= 1 / 3:  # 1X3 以上
            segment_height = width * 3
            segments = self.split_image_vertically(image, segment_height)
        elif aspect_ratio >= 3:  # 3X1 以上
            segment_width = height * 3
            segments = self.split_image_horizontally(image, segment_width)
        else:  # 宽高比合适，直接检测
            segments = [image]

        bboxes = []  # Store images of each bounding box region
        annotated_segments = []

        for segment in segments:
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
                annotated_segment, detection = annotate(segment, boxes=boxes, logits=logits, phrases=phrases)
                annotated_segments.append(annotated_segment)

                # 提取每个边界框区域（分情况过滤）
                for box in detection.xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    width_min = max(160, segment.shape[1] / 5)  # 段宽度的1/5或160px取较大值

                    if is_padded:
                        # 经过padding的图片：宽度≥width_min 且 高度≥160px
                        height_min = 160
                        if bbox_width >= width_min and bbox_height >= height_min:
                            bbox_image = segment[y1:y2, x1:x2]
                            bboxes.append(bbox_image)
                    else:
                        # 未经过padding的图片：保持原逻辑（宽度在 [width_min, 高度] 之间）
                        if width_min <= bbox_width <= bbox_height:
                            bbox_image = segment[y1:y2, x1:x2]
                            bboxes.append(bbox_image)
            except Exception as e:
                logger.error(f"Error during inference: {e}")

        # Combine annotated segments if needed
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

        logger.info(f"Inference completed. {len(bboxes)} bounding boxes detected.")
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
