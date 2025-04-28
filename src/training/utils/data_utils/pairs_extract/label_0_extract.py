import json
from pathlib import Path
import argparse
import random
from collections import defaultdict
from PIL import Image

# ======================
# 可调整参数（集中在开头，支持命令行覆盖）
# ======================
DEEPFASHION_INPUT_DIR = "/Users/sunyuliang/Desktop/AppBuilder/Python/deepfashion_train"  # DeepFashion2原始数据目录
PASS_DATASET_DIR = "/Users/sunyuliang/Desktop/AppBuilder/Python/PASS_dataset"  # PASS数据集路径
PAIRS_OUTPUT_DIR = "/Users/sunyuliang/Desktop/AppBuilder/Python/dinov2_train/Label_0"  # 输出目录
RANDOM_SEED = 52                 # 随机种子
MIN_STYLE = 1                    # 有效style阈值（style>0）
CROP_PADDING = 10                # 裁剪时的边界填充（像素）
MAX_ITEMS_PER_PAIR_ID = 50       # 每个pair_id组内的最大元素数量
MAX_PAIR_ID_OCCURRENCE = 5       # 同一个pair_id的最大出现次数


def parse_arguments():
    parser = argparse.ArgumentParser(description='提取label=0的非衣物与衣物样本对')
    parser.add_argument('--batch_start', type=int, default=1,
                        help='批次起始pair_id（DeepFashion2的pair_id范围）')
    parser.add_argument('--batch_end', type=int, default=100000,
                        help='批次结束pair_id（DeepFashion2的pair_id范围）')
    parser.add_argument('--pairs_to_extract', type=int, default=500,
                        help='目标提取对数')
    return parser.parse_args()


def image_id_from_filename(filename):
    """从文件名提取6位数字ID"""
    return int(filename.stem[:6])  # 截取前6位


def load_valid_clothing_items(args):
    """加载有效衣物item并按category_id分组（过滤style>0）"""
    input_path = Path(DEEPFASHION_INPUT_DIR)
    image_dir = input_path / "image"
    anno_dir = input_path / "annos"

    valid_items = defaultdict(list)  # key: category_id,
    # value: list of (img_id, pair_id, item, img_path, anno_path, original_pair_id)

    for anno_file in anno_dir.glob("*.json"):
        img_id = image_id_from_filename(anno_file)
        if not (args.batch_start <= img_id <= args.batch_end):
            continue

        img_path = image_dir / f"{img_id:06d}.jpg"
        if not img_path.exists():
            continue

        with open(anno_file, 'r') as f:
            data = json.load(f)
        deepfashion_pair_id = data.get("pair_id")  # 原始pair_id
        if not deepfashion_pair_id:
            continue

        for item_key in data:
            if not item_key.startswith("item"):
                continue
            item_data = data[item_key]
            if item_data.get("style", 0) < MIN_STYLE or not item_data.get("bounding_box"):
                continue
            category_id = item_data["category_id"]
            valid_items[category_id].append((
                img_id,           # 图片ID
                deepfashion_pair_id,  # 原始pair_id
                item_data,        # 物品信息
                str(img_path),    # 图片路径
                str(anno_file),   # 标注路径
                deepfashion_pair_id  # 新增原始 pair_id
            ))

    return valid_items


def get_random_non_clothing_image():
    """从PASS数据集中随机获取一张非衣物图片"""
    pass_path = Path(PASS_DATASET_DIR)
    image_files = list(pass_path.glob('**/*.jpg')) + list(pass_path.glob('**/*.png'))
    if not image_files:
        raise ValueError("PASS数据集中未找到图片")
    return random.choice(image_files)


def crop_clothing_image(img_path, bbox, padding=CROP_PADDING):
    """根据边界框裁剪衣物区域"""
    try:
        img = Image.open(img_path)
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img.width, x2 + padding)
        y2 = min(img.height, y2 + padding)
        return img.crop((x1, y1, x2, y2))
    except Exception as e:
        print(f"裁剪失败 {img_path} (边界框{bbox}): {str(e)}")
        return None


def save_pair_metadata(pair_dir, current_pair_num, clothing_info, non_clothing_path):
    """保存优化后的metadata结构"""
    # 解析衣物信息
    img1_id, pair1_id, item1, img1_path, anno1_path, original_pair_id1 = clothing_info

    # 生成新的pair_id和image_id
    new_pair_id = f"pair_{current_pair_num:04d}"
    image1_id = f"{new_pair_id}_1"
    image2_id = f"{new_pair_id}_2"

    metadata = {
        "pair_id": new_pair_id,
        "similarity": 0.0,
        "image1": {
            "image_id": image1_id,
            "original_image_id": img1_id,     # DeepFashion2原始图片ID
            "original_pair_id": original_pair_id1,  # 新增原始 pair_id
            "category_id": item1["category_id"],
            "style": item1["style"],
            "bounding_box": item1["bounding_box"],
            "image_path": str(pair_dir / "image_01.jpg"),
            "original_anno_path": anno1_path
        },
        "image2": {
            "image_id": image2_id,
            "original_image_id": None,
            "original_pair_id": None,
            "category_id": None,
            "style": None,
            "bounding_box": None,
            "image_path": str(pair_dir / "image_02.jpg"),
            "original_anno_path": None
        }
    }
    with open(pair_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)


def save_pairs_to_output(clothing_items, pairs_to_extract, output_dir):
    """保存配对图片和元数据"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 获取当前已经存在的照片对数量
    existing_pairs = list(output_path.glob("pair_*"))
    current_pair_num = len(existing_pairs) + 1

    pair_id_usage_count = defaultdict(int)
    all_clothing_items = [item for sublist in clothing_items.values() for item in sublist]

    while current_pair_num <= len(existing_pairs) + pairs_to_extract:
        clothing_info = random.choice(all_clothing_items)
        pair_id = clothing_info[1]
        if pair_id_usage_count[pair_id] >= MAX_PAIR_ID_OCCURRENCE:
            continue

        non_clothing_path = get_random_non_clothing_image()

        # 裁剪衣物图片
        clothing_img_cropped = crop_clothing_image(clothing_info[3], clothing_info[2]["bounding_box"])
        if not clothing_img_cropped:
            continue

        # 创建对目录
        pair_dir = output_path / f"pair_{current_pair_num:04d}"
        pair_dir.mkdir(exist_ok=True)

        # 保存图片
        clothing_img_cropped.save(pair_dir / "image_01.jpg")
        non_clothing_img = Image.open(non_clothing_path)
        non_clothing_img.save(pair_dir / "image_02.jpg")

        # 保存元数据
        save_pair_metadata(pair_dir, current_pair_num, clothing_info, non_clothing_path)

        print(f"✅ 保存第{current_pair_num:04d}对: {pair_dir}")
        pair_id_usage_count[pair_id] += 1
        current_pair_num += 1


if __name__ == "__main__":
    args = parse_arguments()
    random.seed(RANDOM_SEED)

    # 1. 加载有效衣物item
    print(f"\n=== 加载有效的衣物item（style>0）===")
    clothing_items = load_valid_clothing_items(args)
    if not clothing_items:
        print("❌ 未找到有效衣物item（检查style和category_id）")
        exit(1)

    # 2. 生成非衣物与衣物配对
    print(f"\n=== 生成{args.pairs_to_extract}对label=0的样本（非衣物与衣物）===")

    # 3. 保存配对
    print(f"\n=== 开始保存配对到 {PAIRS_OUTPUT_DIR} ===")
    save_pairs_to_output(clothing_items, args.pairs_to_extract, PAIRS_OUTPUT_DIR)
    print(f"\n🎉 完成！共新增{args.pairs_to_extract}对label=0的相似样本")
