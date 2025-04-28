from dataclasses import dataclass
from typing import List

@dataclass
class ImageInfo:
    image_id: int
    file_path: str
    category_id: int
    style: int
    bounding_box: List[int]
    json_path: str

@dataclass
class ImagePair:
    pair_id: int
    pair_image_id: int
    image1: ImageInfo
    image2: ImageInfo
    similarity: float
