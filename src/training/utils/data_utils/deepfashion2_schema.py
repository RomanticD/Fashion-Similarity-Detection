from dataclasses import dataclass
from typing import List, Dict, Optional

# 1. 类别映射表（完整覆盖13个类别）
CATEGORY_MAPPING: Dict[int, str] = {
    1: "short sleeve top",
    2: "long sleeve top",
    3: "short sleeve outwear",
    4: "long sleeve outwear",
    5: "vest",
    6: "sling",
    7: "shorts",
    8: "trousers",
    9: "skirt",
    10: "short sleeve dress",
    11: "long sleeve dress",
    12: "vest dress",
    13: "sling dress"
}

# 2. 数据类定义
@dataclass
class DeepFashion2Item:
    """单个服装物品的详细标注"""
    category_name: str
    category_id: int
    style: int
    bounding_box: List[int]  # [x1, y1, x2, y2]
    landmarks: Optional[List[int]]  # [x1, y1, v1, ..., xn, yn, vn]
    segmentation: Optional[List[List[float]]]  # 多边形列表
    scale: int  # 1-3
    occlusion: int  # 1-3
    zoom_in: int  # 1-3
    viewpoint: int  # 1-3

@dataclass
class DeepFashion2ImageAnnotation:
    """单张图片的完整标注信息"""
    source: str  # 'shop' 或 'user'
    pair_id: int
    items: Dict[str, DeepFashion2Item]


# 示例JSON解析
import json

json_str = """
{"item2": {"segmentation": [[220.25, 187.55, 259.6, 177.6, 296.6, 158.6, 311.6, 148.2, 327.0, 146.4, 339.4, 144.6, 350.2, 145.2, 379.0, 145.2, 401.8, 148.2, 415.0, 159.0, 427.2, 183.8, 434.4, 212.4, 439.6, 255.6, 445.2, 280.8, 439.2, 292.8, 429.0, 301.0, 417.4, 306.6, 416.8, 324.8, 419.6, 344.8, 423.6, 364.2, 425.4, 380.2, 432.0, 400.0, 434.37, 414.51, 437.06, 428.4, 442.13, 440.2, 442.6, 453.2, 444.0, 476.0, 402.0, 506.0, 352.0, 523.0, 290.02, 531.28, 283.01, 503.18, 274.63, 456.91, 271.37, 432.95, 259.12, 394.55, 256.75, 375.73, 248.2, 355.7, 236.08, 358.71, 225.06, 357.45, 204.98, 348.01, 174.83, 305.67, 172.13, 255.97, 177.0, 221.0, 184.0, 214.0, 196.0, 210.4, 207.2, 199.6, 213.0, 196.8]], "scale": 3, "viewpoint": 3, "zoom_in": 1, "landmarks": [271, 172, 2, 313, 145, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 221, 184, 2, 412, 149, 2, 437, 217, 2, 447, 275, 2, 417, 303, 2, 409, 261, 2, 405, 233, 2, 411, 284, 2, 424, 366, 2, 444, 466, 2, 356, 521, 2, 281, 534, 1, 262, 415, 1, 221, 325, 1, 213, 282, 1, 216, 300, 1, 212, 319, 1, 169, 308, 1, 176, 268, 1, 182, 218, 1], "style": 2, "bounding_box": [165, 133, 466, 572], "category_id": 1, "occlusion": 2, "category_name": "short sleeve top"}, "source": "shop", "pair_id": 1, "item1": {"segmentation": [[145.21, 314.0, 162.67, 312.8, 175.12, 313.05, 205.41, 317.88, 225.8, 331.8, 231.8, 345.8, 246.95, 376.12, 258.85, 392.38, 273.0, 415.0, 287.12, 431.95, 283.0, 446.0, 275.0, 458.0, 264.04, 468.63, 272.26, 502.12, 290.69, 528.35, 303.87, 553.92, 343.32, 597.25, 361.0, 633.57, 346.0, 661.0, 340.2, 672.6, 327.8, 687.6, 323.2, 694.8, 312.34, 701.13, 264.0, 701.0, 182.0, 702.0, 128.0, 701.0, 115.0, 692.0, 113.0, 677.0, 101.0, 654.0, 72.95, 557.0, 40.0, 496.0, 27.0, 469.0, 25.0, 499.0, 21.0, 512.0, 25.0, 540.0, 12.0, 555.0, 16.0, 537.0, 16.0, 502.0, 13.0, 493.0, 2.0, 481.0, 2.0, 472.0, 2.0, 416.0, 2.0, 375.0, 8.0, 349.0, 45.0, 320.0, 77.0, 309.0, 81.0, 342.0, 117.54, 356.24, 149.95, 351.41, 160.76, 338.21, 158.61, 320.22]], "scale": 3, "viewpoint": 2, "zoom_in": 2, "landmarks": [127, 335, 1, 73, 340, 1, 107, 354, 1, 140, 354, 2, 158, 342, 2, 158, 311, 1, 20, 347, 1, 0, 0, 0, 0, 0, 0, 21, 533, 2, 23, 496, 2, 30, 473, 2, 49, 510, 2, 84, 607, 2, 0, 0, 0, 0, 0, 0, 359, 632, 1, 298, 551, 2, 262, 473, 2, 242, 424, 2, 254, 443, 2, 259, 464, 2, 293, 442, 2, 261, 386, 1, 216, 324, 2], "style": 1, "bounding_box": [1, 300, 367, 701], "category_id": 1, "occlusion": 2, "category_name": "short sleeve top"}}
"""

data = json.loads(json_str)
source = data.pop('source')
pair_id = data.pop('pair_id')
items = {k: DeepFashion2Item(**v) for k, v in data.items()}
annotation = DeepFashion2ImageAnnotation(source=source, pair_id=pair_id, items=items)

print(annotation)
