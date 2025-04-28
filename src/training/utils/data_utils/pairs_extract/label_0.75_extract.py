import json
from pathlib import Path
import argparse
import random
from collections import defaultdict
from PIL import Image


# ======================
# 可调整参数（集中在开头，支持命令行覆盖）
# ======================
INPUT_DIR = "/Users/sunyuliang/Desktop/AppBuilder/Python/deepfashion_train"  # 原始数据目录
IMAGE_DIR_NAME = "image"         # 图片文件夹名
ANNO_DIR_NAME = "annos"          # DeepFashion2原始annos文件夹
PAIRS_OUTPUT_DIR = "/Users/sunyuliang/Desktop/AppBuilder/Python/dinov2_train/Label_0.75"  # 输出目录
RANDOM_SEED = 42                 # 随机种子
MIN_STYLE = 1                    # 有效style阈值（style>0）
CROP_PADDING = 10                # 裁剪时的边界填充（像素）
MAX_ITEMS_PER_PAIR_ID = 50        # 每个pair_id组内的最大元素数量
MAX_PAIR_ID_OCCURRENCE = 5       # 同一个pair_id的最大出现次数


def parse_arguments():
    parser = argparse.ArgumentParser(description='提取label=0.75的同类别不同款式对（不同pair_id+同category_id+style>0）')
    parser.add_argument('--batch_start', type=int, default=1,
                        help='批次起始pair_id（DeepFashion2的pair_id范围）')
    parser.add_argument('--batch_end', type=int, default=100000,
                        help='批次结束pair_id（DeepFashion2的pair_id范围）')
    parser.add_argument('--pairs_to_extract', type=int, default=1000,
                        help='目标提取对数')
    return parser.parse_args()


def image_id_from_filename(filename):
    """从文件名提取6位数字ID"""
    return int(filename.stem[:6])  # 截取前6位


def load_valid_items(args):
    """加载有效item并按category_id分组（过滤style>0）"""
    input_path = Path(INPUT_DIR)
    image_dir = input_path / IMAGE_DIR_NAME
    anno_dir = input_path / ANNO_DIR_NAME

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


def generate_similar_pairs(valid_items, pairs_to_extract):
    """生成不同pair_id、同category_id、style>0的相似样本对"""
    similar_pairs = []
    random.seed(RANDOM_SEED)
    pair_id_usage_count = defaultdict(int)  # 记录每个pair_id的使用次数
    category_usage_count = defaultdict(int)  # 记录每个类别的使用次数
    total_categories = len(valid_items)

    # 计算每个类别大致应分配的配对数量
    base_pairs_per_category = pairs_to_extract // total_categories
    remaining_pairs = pairs_to_extract % total_categories

    category_list = list(valid_items.keys())
    random.shuffle(category_list)

    for category_id in category_list:
        items_in_category = valid_items[category_id]
        # 按pair_id分组，避免同pair_id内配对
        pair_groups = defaultdict(list)
        for item in items_in_category:
            if len(pair_groups[item[1]]) < MAX_ITEMS_PER_PAIR_ID:
                pair_groups[item[1]].append(item)  # 按pair_id分组：key=pair_id, value=该pair_id下的所有item

        # 提取所有不同pair_id的组合
        pair_ids = list(pair_groups.keys())
        num_pairs_for_this_category = base_pairs_per_category
        if remaining_pairs > 0:
            num_pairs_for_this_category += 1
            remaining_pairs -= 1

        while num_pairs_for_this_category > 0 and len(pair_ids) >= 2:
            pair_id_i = random.choice(pair_ids)
            pair_ids.remove(pair_id_i)
            pair_id_j = random.choice(pair_ids)

            if pair_id_usage_count[pair_id_i] >= MAX_PAIR_ID_OCCURRENCE or pair_id_usage_count[pair_id_j] >= MAX_PAIR_ID_OCCURRENCE:
                continue

            items_i = pair_groups[pair_id_i]
            items_j = pair_groups[pair_id_j]

            item_i = random.choice(items_i)
            item_j = random.choice(items_j)

            similar_pairs.append((item_i, item_j))
            pair_id_usage_count[pair_id_i] += 1
            pair_id_usage_count[pair_id_j] += 1
            category_usage_count[category_id] += 1
            num_pairs_for_this_category -= 1

            if len(similar_pairs) >= pairs_to_extract:
                break

        if len(similar_pairs) >= pairs_to_extract:
            break

    random.shuffle(similar_pairs)
    return similar_pairs[:pairs_to_extract]


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


def save_pair_metadata(pair_dir, current_pair_num, img1_info, img2_info):
    """保存优化后的metadata结构"""
    # 解析原始信息
    img1_id, pair1_id, item1, img1_path, anno1_path, original_pair_id1 = img1_info
    img2_id, pair2_id, item2, img2_path, anno2_path, original_pair_id2 = img2_info

    # 生成新的pair_id和image_id
    new_pair_id = f"pair_{current_pair_num:04d}"
    image1_id = f"{new_pair_id}_1"
    image2_id = f"{new_pair_id}_2"

    metadata = {
        "pair_id": new_pair_id,
        "similarity": 0.75,
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
            "original_image_id": img2_id,
            "original_pair_id": original_pair_id2,  # 新增原始 pair_id
            "category_id": item2["category_id"],
            "style": item2["style"],
            "bounding_box": item2["bounding_box"],
            "image_path": str(pair_dir / "image_02.jpg"),
            "original_anno_path": anno2_path
        }
    }
    with open(pair_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)


def save_pairs_to_output(pairs, output_dir):
    """保存配对图片和元数据"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    current_pair_num = 1

    for img1_info, img2_info in pairs:
        # 解析信息
        img1_id, pair1_id, item1, img1_path, anno1_path, original_pair_id1 = img1_info
        img2_id, pair2_id, item2, img2_path, anno2_path, original_pair_id2 = img2_info

        # 裁剪图片
        img1_cropped = crop_clothing_image(img1_path, item1["bounding_box"])
        img2_cropped = crop_clothing_image(img2_path, item2["bounding_box"])
        if not img1_cropped or not img2_cropped:
            continue

        # 创建对目录
        pair_dir = output_path / f"pair_{current_pair_num:04d}"
        pair_dir.mkdir(exist_ok=True)

        # 保存图片
        img1_cropped.save(pair_dir / "image_01.jpg")
        img2_cropped.save(pair_dir / "image_02.jpg")

        # 保存元数据
        save_pair_metadata(pair_dir, current_pair_num, img1_info, img2_info)

        print(f"✅ 保存第{current_pair_num:04d}对: {pair_dir}")
        current_pair_num += 1


if __name__ == "__main__":
    args = parse_arguments()
    random.seed(RANDOM_SEED)

    # 1. 加载有效item（按category_id分组，过滤style>0）
    print(f"\n=== 加载category_id有效的item（style>0）===")
    valid_items = load_valid_items(args)
    if not valid_items:
        print("❌ 未找到有效item（检查style和category_id）")
        exit(1)

    # 2. 生成跨pair_id的同category_id配对
    print(f"\n=== 生成{args.pairs_to_extract}对label=0.75的样本（不同pair_id+同category_id）===")
    similar_pairs = generate_similar_pairs(valid_items, args.pairs_to_extract)
    if not similar_pairs:
        print(f"❌ 未找到符合条件的配对（检查category_id分布）")
        exit(1)

    # 3. 保存配对
    print(f"\n=== 开始保存配对到 {PAIRS_OUTPUT_DIR} ===")
    save_pairs_to_output(similar_pairs, PAIRS_OUTPUT_DIR)
    print(f"\n🎉 完成！共保存{len(similar_pairs)}对label=0.75的相似样本")
