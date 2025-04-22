from .image_similarity_resnet50 import ImageSimilarityResNet50
from .image_similarity_DINOv2 import ImageSimilarityDINOv2
from .image_similarity_vit import ImageSimilarityViT


class ImageSimilarityIndex:
    def __init__(self):
        self.models = {
            'resnet50': ImageSimilarityResNet50(),
            'DINOv2': ImageSimilarityDINOv2(),
            'ViT': ImageSimilarityViT()
        }

    def get_model(self, model_name):
        if model_name not in self.models:
            raise ValueError(f"Unsupported model: {model_name}")
        return self.models[model_name]
