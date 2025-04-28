# src/core/image_upload.py
import base64
import json
import numpy as np
from pathlib import Path
from PIL import Image
import logging
import traceback
import os

from src.repo.split_images_repo import save_to_db
from src.utils.data_conversion import numpy_to_base64
# 使用微调后的模型
from src.core.image_similarity import ImageSimilarity

# 配置日志
logger = logging.getLogger(__name__)

class ImageUploader:
    """
    A class for uploading processed images to the database.
    """

    def __init__(self):
        """
        Initialize the image uploader with the fine-tuned model.
        """
        # 使用全局定义的微调模型实例
        self.similarity_model = ImageSimilarity
        logger.info("ImageUploader初始化完成，使用微调DINOv2模型")

    def upload_splitted_image_to_db(self, image_data: np.ndarray, splitted_image_id: str,
                                    splitted_image_path: str, original_image_id: str,
                                    bounding_box: str, image_format: str, vector: str):
        """
        Upload a segmented image and its feature vector to the database.

        Args:
            image_data (np.ndarray): The image data as a numpy array.
            splitted_image_id (str): The ID for the segmented image.
            splitted_image_path (str): The path where the image is saved.
            original_image_id (str): The ID of the original image.
            bounding_box (str): The bounding box coordinates.
            image_format (str): The image format (e.g., 'PNG', 'JPEG').
            vector (str): The feature vector.
        """
        try:
            logger.info(f"开始上传分割图像到数据库: {splitted_image_id}")
            
            # 验证图像数据
            if image_data is None or image_data.size == 0:
                logger.error(f"图像数据为空: {splitted_image_id}")
                return False
                
            # 验证向量数据
            if vector is None:
                logger.error(f"特征向量为空: {splitted_image_id}")
                return False
            
            # Convert image to Base64
            base64_image = numpy_to_base64(image_data, image_format)

            # Decode to binary data
            binary_data = base64.b64decode(base64_image)

            # Convert vector to JSON string
            vector_json = json.dumps(vector.tolist())
            
            logger.debug(f"图像 {splitted_image_id} 转换完成，图像大小: {len(binary_data)} 字节, 向量维度: {len(vector.tolist())}")

            # Save to database
            save_result = save_to_db(splitted_image_id, splitted_image_path, original_image_id,
                   bounding_box, binary_data, vector_json)
                   
            if save_result:
                logger.info(f"成功保存图像到数据库: {splitted_image_id}")
                return True
            else:
                logger.warning(f"保存图像到数据库失败: {splitted_image_id}")
                return False
                
        except Exception as e:
            logger.error(f"上传图像到数据库时出错: {e}")
            logger.debug(traceback.format_exc())
            return False

    def process_and_upload_image(self, image_array: np.ndarray, idx: int,
                                 save_path: Path, image_name: str):
        """
        Process an image and upload it to the database.

        Args:
            image_array (np.ndarray): The image array.
            idx (int): The image index.
            save_path (Path): Where to save the image.
            image_name (str): The original image name.

        Returns:
            Path: The path where the image was saved, or None if failed.
        """
        try:
            logger.info(f"开始处理图像: {image_name}/segment_{idx}.png")
            
            # 检查图像数组
            if image_array is None or image_array.size == 0:
                logger.error(f"图像数组为空: {image_name}/segment_{idx}.png")
                return None
                
            # 检查图像维度和通道
            logger.debug(f"图像形状: {image_array.shape}")
            if len(image_array.shape) < 3 or image_array.shape[2] < 3:
                logger.warning(f"图像 {image_name}/segment_{idx}.png 不是RGB格式, 形状: {image_array.shape}")
                # 尝试将灰度图转为RGB
                if len(image_array.shape) == 2:
                    logger.info(f"将灰度图转换为RGB: {image_name}/segment_{idx}.png")
                    rgb_array = np.stack((image_array,) * 3, axis=-1)
                    image_array = rgb_array
            
            # 确保图像数据类型为uint8
            if image_array.dtype != np.uint8:
                logger.warning(f"图像数据类型非标准: {image_array.dtype}，转换为uint8")
                if np.issubdtype(image_array.dtype, np.floating):
                    image_array = (image_array * 255).astype(np.uint8)
                else:
                    image_array = image_array.astype(np.uint8)
            
            # Extract image features using the fine-tuned model
            try:
                logger.info(f"从图像提取特征向量: {image_name}/segment_{idx}.png")
                vector = self.similarity_model.extract_feature(image_array)
                logger.info(f"特征向量提取成功，维度: {vector.shape}")
            except Exception as e:
                logger.error(f"特征向量提取失败: {e}")
                logger.debug(traceback.format_exc())
                return None

            # Save image to file system
            try:
                logger.info(f"保存图像到文件系统: {save_path}")
                img = Image.fromarray(image_array)
                
                # 确保目录存在
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                # 保存为PNG格式
                img.save(save_path, format='PNG')
                logger.info(f"图像成功保存到: {save_path}")
            except Exception as e:
                logger.error(f"保存图像到文件系统失败: {e}")
                logger.debug(traceback.format_exc())
                return None

            # Create relative path format
            relative_path = f"{image_name}/segment_{idx}.png"

            # Create unique ID including original image name
            unique_splitted_image_id = f"{image_name}_segment_{idx}"

            # Upload to database
            upload_success = self.upload_splitted_image_to_db(
                image_data=image_array,
                splitted_image_id=unique_splitted_image_id,
                splitted_image_path=relative_path,
                original_image_id=image_name,
                bounding_box=str(idx),
                image_format="PNG",  # 明确指定为PNG格式
                vector=vector
            )
            
            if upload_success:
                logger.info(f"图像处理和上传成功: {unique_splitted_image_id}")
                return save_path
            else:
                logger.warning(f"图像上传到数据库失败: {unique_splitted_image_id}")
                return None
                
        except Exception as e:
            logger.error(f"处理和上传图像时出错: {image_name}/segment_{idx}.png, 错误: {e}")
            logger.debug(traceback.format_exc())
            return None
