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
PAIRS_OUTPUT_DIR = "/Users/sunyuliang/Desktop/AppBuilder/Python/dinov2_train/Label_1"  # 输出目录
RANDOM_SEED = 4                  # 随机种子
MIN_STYLE = 1                    # 有效style阈值（style>0）
CROP_PADDING = 10                # 裁剪时的边界填充（像素）


def parse_arguments():
    parser = argparse.ArgumentParser(description='提取label=1的完全同款对（支持续生成）')
    parser.add_argument('--batch_start', type=int, default=1,
                        help='批次起始pair_id（DeepFashion2的pair_id范围）')
    parser.add_argument('--batch_end', type=int, default=100000,
                        help='批次结束pair_id（DeepFashion2的pair_id范围）')
    parser.add_argument('--pairs_to_extract', type=int, default=1000,
                        help='目标新增对数（总对数=已有对+新增对）')
    return parser.parse_args()


def image_id_from_filename(filename):
    """从文件名提取6位数字ID"""
    return int(filename.stem[:6])  # 截取前6位


def load_valid_items(args):
    """加载有效item并关联原始annos路径"""
    input_path = Path(INPUT_DIR)
    image_dir = input_path / IMAGE_DIR_NAME
    anno_dir = input_path / ANNO_DIR_NAME

    valid_items = defaultdict(list)  # key: (deepfashion_pair_id, style),
    # value: list of (img_id, item, img_path, anno_path, original_pair_id)

    for anno_file in anno_dir.glob("*.json"):
        img_id = image_id_from_filename(anno_file)
        if not (args.batch_start <= img_id <= args.batch_end):
            continue

        img_path = image_dir / f"{img_id:06d}.jpg"
        if not img_path.exists():
            continue

        with open(anno_file, 'r') as f:
            data = json.load(f)
        deepfashion_pair_id = data.get("pair_id")  # 原始pair_id（来自DeepFashion2）
        if not deepfashion_pair_id:
            continue

        for item_key in data:
            if not item_key.startswith("item"):
                continue
            item_data = data[item_key]
            if item_data.get("style", 0) < MIN_STYLE or not item_data.get("bounding_box"):
                continue
            valid_items[(deepfashion_pair_id, item_data["style"])].append((
                img_id,
                item_data,
                str(img_path),
                str(anno_file),  # 原始annos路径
                deepfashion_pair_id  # 新增原 pair_id
            ))

    return valid_items


def generate_positive_pairs(valid_items, pairs_to_extract):
    """生成同pair_id、同style的跨图片正样本对"""
    positive_pairs = []
    random.seed(RANDOM_SEED)

    for (deepfashion_pair_id, style), items_in_style in valid_items.items():
        img_groups = defaultdict(list)
        for img_id, item, img_path, anno_path, original_pair_id in items_in_style:
            img_groups[img_id].append((item, img_path, anno_path, original_pair_id))

        img_ids = list(img_groups.keys())
        for i in range(len(img_ids)):
            for j in range(i + 1, len(img_ids)):
                for item1, path1, anno1, original_pair_id1 in img_groups[img_ids[i]]:
                    for item2, path2, anno2, original_pair_id2 in img_groups[img_ids[j]]:
                        positive_pairs.append((
                            (img_ids[i], item1, path1, anno1, original_pair_id1),
                            (img_ids[j], item2, path2, anno2, original_pair_id2)
                        ))

    random.shuffle(positive_pairs)
    return positive_pairs[:pairs_to_extract]


def crop_clothing_image(img_path, bbox, padding=CROP_PADDING):
    """根据边界框裁剪衣物区域"""
    try:
        img = Image.open(img_path)  # 打开图片路径
        x1, y1, x2, y2 = bbox  # 解析边界框
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img.width, x2 + padding)
        y2 = min(img.height, y2 + padding)
        return img.crop((x1, y1, x2, y2))
    except Exception as e:
        print(f"裁剪失败 {img_path} (边界框{bbox}): {str(e)}")  # 增强错误提示
        return None


def save_pair_metadata(pair_dir, current_pair_num, img1_info, img2_info):
    """保存优化后的metadata结构"""
    img1_id, item1, img1_path, img1_anno, original_pair_id1 = img1_info
    img2_id, item2, img2_path, img2_anno, original_pair_id2 = img2_info

    # 生成新的pair_id和image_id
    pair_id = f"pair_{current_pair_num:04d}"
    image1_id = f"{pair_id}_1"
    image2_id = f"{pair_id}_2"

    metadata = {
        "pair_id": pair_id,
        "similarity": 1.0,
        "image1": {
            "original_pair_id": original_pair_id1,  # 新增原 pair_id
            "image_id": image1_id,
            "original_id": img1_id,  # DeepFashion2原始图片ID
            "category_id": item1["category_id"],
            "style": item1["style"],
            "bounding_box": item1["bounding_box"],
            "image_path": str(pair_dir / "image_01.jpg"),  # 完整路径
            "original_anno_path": img1_anno  # 重命名字段
        },
        "image2": {
            "original_pair_id": original_pair_id2,  # 新增原 pair_id
            "image_id": image2_id,
            "original_id": img2_id,
            "category_id": item2["category_id"],
            "style": item2["style"],
            "bounding_box": item2["bounding_box"],
            "image_path": str(pair_dir / "image_02.jpg"),
            "original_anno_path": img2_anno
        }
    }
    with open(pair_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)


def save_pairs_to_output(pairs, output_dir):
    """保存配对图片和元数据（支持续生成）"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 获取已有对数量并计算起始编号
    existing_pairs = list(output_path.glob("pair_*.json"))  # 通过json文件判断已有对
    current_pair_num = len(existing_pairs) + 1  # 从下一个编号开始

    for img1_info, img2_info in pairs:
        # 解析信息（结构：(img_id, item, img_path, anno_path, original_pair_id)）
        img1_id, item1, img1_path, img1_anno, original_pair_id1 = img1_info
        img2_id, item2, img2_path, img2_anno, original_pair_id2 = img2_info

        # 裁剪图片（参数顺序：图片路径 -> 边界框）
        img1_cropped = crop_clothing_image(img1_path, item1["bounding_box"])  # 修复此处
        img2_cropped = crop_clothing_image(img2_path, item2["bounding_box"])  # 修复此处
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

    # 1. 加载有效item（含原始annos路径）
    print(f"\n=== 加载DeepFashion2 pair_id {args.batch_start}-{args.batch_end} 的有效item ===")
    valid_items = load_valid_items(args)
    if not valid_items:
        print("❌ 未找到有效item（检查style和bounding_box）")
        exit(1)

    # 2. 生成正样本对（生成数量为目标新增数）
    print(f"\n=== 生成{args.pairs_to_extract}对label=1的样本 ===")
    positive_pairs = generate_positive_pairs(valid_items, args.pairs_to_extract)
    if not positive_pairs:
        print(f"❌ 未找到符合条件的配对（检查pair_id分布）")
        exit(1)

    # 3. 保存配对（从已有对后继续生成）
    print(f"\n=== 开始保存配对到 {PAIRS_OUTPUT_DIR} ===")
    save_pairs_to_output(positive_pairs, PAIRS_OUTPUT_DIR)
    print(f"\n🎉 完成！共新增{len(positive_pairs)}对label=1的正样本，当前总对数{len(list(Path(PAIRS_OUTPUT_DIR).glob('pair_*')))}")
