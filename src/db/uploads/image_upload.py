# src/core/image_upload.py
import base64
import json
import numpy as np
from pathlib import Path
from PIL import Image

from src.repo.split_images_repo import save_to_db
from src.utils.data_conversion import numpy_to_base64
# 使用微调后的模型
from src.core.image_similarity import ImageSimilarity


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
        # Convert image to Base64
        base64_image = numpy_to_base64(image_data, image_format)

        # Decode to binary data
        binary_data = base64.b64decode(base64_image)

        # Convert vector to JSON string
        vector_json = json.dumps(vector.tolist())

        # Save to database
        save_to_db(splitted_image_id, splitted_image_path, original_image_id,
                   bounding_box, binary_data, vector_json)

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
            Path: The path where the image was saved.
        """
        # Extract image features using the fine-tuned model
        vector = self.similarity_model.extract_feature(image_array)

        # Save image to file system
        img = Image.fromarray(image_array)
        img.save(save_path)

        # Create relative path format
        relative_path = f"{image_name}/segment_{idx}.png"

        # Create unique ID including original image name
        unique_splitted_image_id = f"{image_name}_segment_{idx}"

        # Upload to database
        self.upload_splitted_image_to_db(
            image_data=image_array,
            splitted_image_id=unique_splitted_image_id,
            splitted_image_path=relative_path,
            original_image_id=image_name,
            bounding_box=str(idx),
            image_format="png",
            vector=vector
        )

        return save_path
