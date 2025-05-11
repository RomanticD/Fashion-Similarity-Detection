import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import argparse
from src.db.uploads.image_upload import ImageUploader

DEEPFASHION_DIR = Path("/Users/sunyuliang/Desktop/AppBuilder/Python/deepfashion_train")
NUM_IMAGES = 1
CROP_PADDING = 10
MIN_STYLE = 1
RANDOM_SEED = 42


def load_valid_items():
    """加载符合要求的衣物item（基于label_1_extract的模式）"""
    valid_items = []
    image_dir = DEEPFASHION_DIR / "image"
    anno_dir = DEEPFASHION_DIR / "annos"
    
    for anno_file in anno_dir.glob("*.json"):
        with open(anno_file, 'r') as f:
            data = json.load(f)
        
        pair_id = data.get("pair_id")
        if not pair_id:
            continue
            
        img_id = int(anno_file.stem[:6])
        img_path = image_dir / f"{img_id:06d}.jpg"
        
        for item_key in data:
            if item_key.startswith("item"):
                item = data[item_key]
                # 新增item_id提取逻辑
                if item.get("style", 0) >= MIN_STYLE and item.get("bounding_box"):
                    valid_items.append({
                        "img_id": img_id,
                        "pair_id": pair_id,
                        "item_id": item_key.replace("item", ""),  # 提取item数字ID
                        "item_data": item,
                        "img_path": str(img_path),
                        "anno_path": str(anno_file)
                    })
    return valid_items


def crop_clothing_image(img_path, bbox):
    """基于label_1_extract的裁剪逻辑"""
    try:
        img = Image.open(img_path)
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1 - CROP_PADDING)
        y1 = max(0, y1 - CROP_PADDING)
        x2 = min(img.width, x2 + CROP_PADDING)
        y2 = min(img.height, y2 + CROP_PADDING)
        return img.crop((x1, y1, x2, y2))
    except Exception as e:
        print(f"裁剪失败 {img_path}: {str(e)}")
        return None

# ======================
# 参数结构调整（遵循batch_upload模式）
# ======================
CONFIG = {
    "deepfashion_root": "/path/to/deepfashion",
    "num_samples": 1000,
    "min_style_score": 0.85,  # 更严格的风格筛选
    "crop_margin": 15,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}


def process_and_save(image_data):
    try:
        uploader = ImageUploader()
        
        cropped_img = crop_clothing_image(
            image_data["img_path"],
            image_data["item_data"]["bounding_box"]
        )
        if not cropped_img:
            return False

        # 修正特征处理流程（兼容不同设备）
        feature_tensor = uploader.similarity_model.extract_feature(cropped_img)
        
        # 添加向量有效性校验
        if len(feature_tensor) == 0:
            print("错误：空特征向量")
            return False
            
        vector_str = json.dumps(feature_tensor.tolist())
        
        if vector_str == '[]' or not vector_str:
            print(f"空特征向量: {image_data['img_id']}")
            return False
            
        return uploader.upload_splitted_image_to_db(
            image_data=np.array(cropped_img.convert('RGB')),
            splitted_image_id=f"df2_{image_data['img_id']}_{image_data['item_id']}",
            splitted_image_path=f"/deepfashion2/{image_data['img_id']}_{image_data['item_id']}.jpg",
            original_image_id=f"original_{image_data['img_id']}",
            bounding_box=",".join(map(str, image_data["item_data"]["bounding_box"])),
            image_format="PNG",
            vector=vector_str  # 传递完整的JSON字符串
        )
        
    except Exception as e:
        print(f"上传失败: {str(e)}")
        return False

# 移除冗余的upload_processed_image方法


def main():
    # 加载有效数据
    valid_items = load_valid_items()
    print(f"找到{len(valid_items)}个有效衣物item")
    
    # 随机抽样（添加抽样数量校验）
    selected = random.sample(valid_items, min(args.num, len(valid_items)))
    
    # 处理并保存
    success_count = 0
    for item in selected:
        if process_and_save(item):
            success_count += 1
            
    print(f"完成处理，成功上传{success_count}张图片")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=NUM_IMAGES, help='要处理的图片数量')
    args = parser.parse_args()
    
    # 初始化随机种子
    random.seed(RANDOM_SEED)
    
    main()
    