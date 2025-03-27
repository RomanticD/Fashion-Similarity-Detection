# src/core/image_similarity.py
import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
import time
import tempfile


class ImageSimilarity:
    """
    A class for extracting features from images and comparing their similarities.
    """

    def __init__(self):
        """
        Initialize the image similarity analyzer with a pre-trained model.
        """
        # Load pre-trained ResNet50 model
        self.weights = ResNet50_Weights.IMAGENET1K_V1
        self.model = models.resnet50(weights=self.weights)
        self.model.eval()
        self.model.fc = torch.nn.Identity()  # Keep 2048-dim features

        # Define preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract_feature(self, img_input):
        """
        Extract features from an image.

        Args:
            img_input: Input image (file path, PIL image, or numpy array).

        Returns:
            np.ndarray: The extracted feature vector.

        Raises:
            ValueError: If the input type is not supported.
        """
        start_time = time.time()

        # Handle different input types
        if isinstance(img_input, (str, Path)):  # File path
            img = Image.open(img_input).convert('RGB')
        elif isinstance(img_input, Image.Image):  # PIL image
            img = img_input
        elif isinstance(img_input, np.ndarray):  # NumPy array
            img = Image.fromarray(img_input)
        else:
            raise ValueError("Unsupported input type")

        # Preprocess and extract features
        img_t = self.transform(img).unsqueeze(0)

        with torch.no_grad():
            feat = self.model(img_t)

        feature = feat.squeeze(0).numpy()
        print(f"Feature extraction time: {time.time() - start_time:.4f}s")
        return feature

    def load_images_from_arrays(self, image_arrays):
        """
        Load features from in-memory image arrays.

        Args:
            image_arrays (list): A list of numpy arrays representing images.

        Returns:
            dict: A dictionary mapping segment names to feature vectors.
        """
        return {f"segment_{i}": self.extract_feature(arr) for i, arr in enumerate(image_arrays)}

    def load_single_image_feature_vector(self, img_path):
        """
        Load the feature vector of a single image.

        Args:
            img_path (str or Path): The file path of the image.

        Returns:
            dict: A dictionary mapping the image name to its feature vector.
        """
        return {Path(img_path).name: self.extract_feature(img_path)}

    @staticmethod
    def cosine_similarity(vec1, vec2):
        """
        Calculate the cosine similarity between two vectors.

        Args:
            vec1 (np.ndarray): The first vector.
            vec2 (np.ndarray): The second vector.

        Returns:
            float: The cosine similarity value.
        """
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def compare_similarities(self, single_dict, images_dict):
        """
        Compare the similarity between a single image and multiple images.

        Args:
            single_dict (dict): A dictionary with the feature vector of a single image.
            images_dict (dict): A dictionary with feature vectors of multiple images.

        Returns:
            list: A list of tuples (image_name, similarity_score).
        """
        single_name, single_vec = list(single_dict.items())[0]
        return [(name, self.cosine_similarity(single_vec, vec)) for name, vec in images_dict.items()]