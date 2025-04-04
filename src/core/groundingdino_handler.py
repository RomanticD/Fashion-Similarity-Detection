# src/core/groundingdino_handler.py
import sys
import os
from pathlib import Path
from PIL import Image
import numpy as np

from src.core.image_similarity import ImageSimilarity
from src.core.image_processing import ImageProcessor

# Set up Python path
root_dir = Path(__file__).parent.resolve()
while not (root_dir / "setup.py").exists() and not (root_dir / ".git").exists():
    root_dir = root_dir.parent

groundingdino_path = root_dir / 'GroundingDINO'
sys.path.append(str(groundingdino_path))

# Import necessary libraries
from GroundingDINO.groundingdino.util.inference import load_model


class ClothingDetector:
    """
    A class responsible for detecting clothing items in images using the GroundingDINO model.
    """

    def __init__(self, config_path=None, weights_path=None, box_threshold=0.3, text_prompt="clothes"):
        """
        Initialize the clothing detector with model paths.

        Args:
            config_path (Path, optional): Path to the model configuration file.
            weights_path (Path, optional): Path to the model weights file.
        """
        # Set up paths
        self.root_dir = root_dir

        # Model Backbone
        self.config_path = config_path or groundingdino_path / 'groundingdino' / 'config' / 'GroundingDINO_SwinT_OGC.py'
        self.weights_path = weights_path or root_dir / 'src' / 'checkpoints' / 'groundingdino_swint_ogc.pth'

        # Default parameters
        self.box_threshold = box_threshold
        self.text_prompt = text_prompt

        # Image processor
        self.image_processor = ImageProcessor()

        # Validate weights file
        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Weights file not found. Please ensure '{self.weights_path}' exists in the project root.")

    def load_model(self):
        """
        Load the GroundingDINO model.

        Returns:
            The loaded model.
        """
        return load_model(str(self.config_path), str(self.weights_path), device='cpu')

    def detect_clothes(self, image, frame_window=None, box_threshold=None, text_prompt=None):
        """
        Detect clothing regions in the input image.

        Args:
            image (np.ndarray): The input image as a numpy array.
            frame_window (optional): A window object to display processed images.

        Returns:
            list: A list of bounding boxes for detected clothing regions.
        """
        # Update box_threshold and text_prompt if provided
        if box_threshold is not None:
            self.box_threshold = box_threshold
        if text_prompt is not None:
            self.text_prompt = text_prompt

        # Load model
        model = self.load_model()

        # Get transform function
        transform = self.image_processor.prepare_transform()

        try:
            # Set segment height to 3x image width
            segment_height = image.shape[1] * 3
            segments = self.image_processor.split_image_vertically(image, segment_height)

            # Detect clothing
            clothes_bboxes = self.image_processor.run_inference(
                model, transform, segments, self.text_prompt, self.box_threshold, frame_window
            )

            return clothes_bboxes
        except Exception as e:
            print(f"Error during detection: {e}")
            return []


class DirectoryManager:
    """
    A utility class for managing directories and files.
    """

    @staticmethod
    def clear_directory(data_dir):
        """
        Clear all files and empty folders within a specified directory.

        Args:
            data_dir (Path): The path of the directory to be cleared.
        """
        # Ensure the path is a directory
        if data_dir.exists() and data_dir.is_dir():
            # Traverse all files in the directory and delete them
            for file in data_dir.rglob('*'):
                if file.is_file():
                    file.unlink()  # Delete file
                elif file.is_dir():
                    file.rmdir()  # Delete empty folder
        else:
            print(f"{data_dir} is not a valid directory.")

# Example usage
if __name__ == "__main__":
    # Root directory
    root_dir = Path(__file__).parent.resolve()
    while not (root_dir / "setup.py").exists() and not (root_dir / ".git").exists():
        root_dir = root_dir.parent

    # Check for ScreenShots folder
    screenshots_dir = root_dir / "ScreenShots"
    if not screenshots_dir.exists() or not screenshots_dir.is_dir():
        raise FileNotFoundError(f"'{screenshots_dir}' folder does not exist. Please download resource files.")

    # Image path
    image_path = screenshots_dir / "01.jpeg"
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # Check if image file exists
    if not image_path.exists():
        raise FileNotFoundError(f"Image file '{image_path}' does not exist. Please check the file path.")

    # Open image and convert to numpy array
    image = Image.open(image_path)
    image_np = np.array(image)

    # Initialize the detector and detect clothes
    detector = ClothingDetector()
    result_bboxes = detector.detect_clothes(image_np)

    # Create data directory and process images
    data_dir = root_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectory for the original image
    image_dir = data_dir / image_name
    image_dir.mkdir(parents=True, exist_ok=True)

    # Save segmented images
    saved_paths = []
    for idx, img_array in enumerate(result_bboxes):
        filename = f"segment_{idx}.png"
        save_path = image_dir / filename
        img = Image.fromarray(img_array)
        img.save(save_path)
        saved_paths.append(save_path)
        print(f"Saved segmented image: {save_path}")

    # Load and compare features
    similarity = ImageSimilarity()

    # Load segmented image features
    segmented_features = {}
    for path in saved_paths:
        feature_dict = similarity.load_single_image_feature_vector(path)
        segmented_features.update(feature_dict)

    # Load comparison image feature
    single_img_path = root_dir / "Assets" / "spilt_image_similarity_test_2.png"
    single_feature = similarity.load_single_image_feature_vector(single_img_path)

    # Compare similarities
    similarity_results = similarity.compare_similarities(single_feature, segmented_features)

    # Print comparison results
    print(f"\nComparison image: {single_img_path.name}")
    for img_name, sim in sorted(similarity_results, key=lambda x: x[1], reverse=True):
        print(f"Segmented region {img_name}: Similarity {sim:.4f}")