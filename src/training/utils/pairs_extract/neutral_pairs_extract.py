# Fashion-Similarity-Detection/training/utils/neutral_pairs_extract.py
import json
import shutil
from pathlib import Path
import argparse
import random

# ======================
# 可调整参数（集中在开头，支持命令行覆盖）
# ======================
INPUT_DIR = "/Users/sunyuliang/Desktop/AppBuilder/Python/deepfashion_train"  # 原始数据目录
IMAGE_DIR_NAME = "image"         # 图片文件夹名（如"image"或"images"）
ANNO_DIR_NAME = "annos"          # 标注文件夹名
OUTPUT_BASE_DIR = "/Users/sunyuliang/Desktop/AppBuilder/Python/dinov2_train/training_pairs"  # 输出基目录
RANDOM_SEED = 42                 # 随机种子（固定以确保可复现性）
MIN_ID_DIFF = 50                 # 最小ID差异（中立对要求ID差绝对值≥此值）
MAX_TRIES_PER_PAIR = 1000        # 每对最多尝试次数（避免陷入死循环）


def parse_arguments():
    parser = argparse.ArgumentParser(description='高效随机抽取中立样本对（避免全量遍历）')
    parser.add_argument('--batch_start', type=int, default=100001,
                        help='批次起始数值ID（如50000对应050000.jpg）')
    parser.add_argument('--batch_end', type=int, default=190000,
                        help='批次结束数值ID（如60000对应060000.jpg）')
    parser.add_argument('--pairs_to_extract', type=int, default=495,
                        help='本次提取的中立样本对数')
    return parser.parse_args()


def image_id_from_filename(filename):
    """从Path对象或文件名中提取数值ID（处理jpg/png等格式）"""
    if isinstance(filename, Path):
        stem = filename.stem  # Path对象直接取stem（如"149205"）
    else:
        stem = filename.split('.')[0]  # 兼容字符串情况（如"149205.jpg"）
    return int(stem[:6])  # 截取前6位转为整数


def load_image_with_pairid(args):
    """加载指定ID范围内的所有有效图片及其pair_id（带缓存优化）"""
    input_path = Path(INPUT_DIR)
    image_dir = input_path / IMAGE_DIR_NAME
    anno_dir = input_path / ANNO_DIR_NAME

    valid_images = []
    id_range = range(args.batch_start, args.batch_end + 1)

    # 缓存标注文件存在性检查结果
    anno_cache = {}
    for img_id in id_range:
        anno_file = anno_dir / f"{img_id:06d}.json"
        anno_cache[img_id] = anno_file.exists()

    for img_file in image_dir.glob("*.jpg"):  # img_file是Path对象
        try:
            # 直接使用Path对象的stem属性获取文件名前缀（如"149205"）
            img_id = image_id_from_filename(img_file)  # 传入Path对象
            if img_id not in id_range:
                continue

            if not anno_cache.get(img_id, False):
                print(f"警告：跳过无标注文件的图片 {img_file.name}")
                continue

            with open(anno_dir / f"{img_id:06d}.json", 'r') as f:
                pair_id = json.load(f).get("pair_id")

            if not pair_id:
                print(f"警告：跳过无pair_id的图片 {img_file.name}")
                continue

            valid_images.append((img_id, str(img_file), pair_id))
        except Exception as e:
            print(f"处理图片 {img_file.name} 失败: {str(e)}")
            continue

    if not valid_images:
        raise ValueError(f"在ID范围 {args.batch_start}-{args.batch_end} 内未找到有效图片")

    return valid_images


def random_sample_pairs(valid_images, num_pairs, min_id_diff=MIN_ID_DIFF, max_tries=MAX_TRIES_PER_PAIR):
    """高效随机采样满足条件的配对（避免O(n²)复杂度）"""
    random.seed(RANDOM_SEED)
    pairs = []
    tried_pairs = set()  # 记录已尝试的组合避免重复检查

    while len(pairs) < num_pairs and max_tries > 0:
        # 随机选择两张不同的图片
        idx1, idx2 = random.sample(range(len(valid_images)), 2)
        img1 = valid_images[idx1]
        img2 = valid_images[idx2]

        # 检查是否已尝试过该组合
        key = tuple(sorted((idx1, idx2)))
        if key in tried_pairs:
            continue
        tried_pairs.add(key)

        # 检查条件：pair_id不同且ID差≥min_id_diff
        if img1[2] != img2[2] and abs(img1[0] - img2[0]) >= min_id_diff:
            pairs.append((img1[1], img2[1]))
            max_tries = MAX_TRIES_PER_PAIR  # 成功找到时重置尝试次数
        else:
            max_tries -= 1  # 失败时减少尝试次数

        # 防止无限循环
        if max_tries <= 0:
            raise RuntimeError("达到最大尝试次数仍未找到足够配对，可能数据分布不符合条件")

    return pairs


def get_next_pair_number(output_dir):
    """获取当前输出目录的最大对编号，支持不连续编号"""
    existing_pairs = [d.name for d in Path(output_dir).glob("pair_????") if d.is_dir()]
    return 1 if not existing_pairs else max(int(d.split('_')[1]) for d in existing_pairs) + 1


def save_pairs_to_output(pairs, output_dir, start_number):
    """按动态序号保存配对，延续人工筛选后的编号"""
    output_path = Path(output_dir)
    current_number = start_number
    saved_pairs = 0

    for img1, img2 in pairs:
        if saved_pairs >= len(pairs):
            break
        pair_dir = output_path / f"pair_{current_number:04d}"
        if pair_dir.exists():
            print(f"警告：pair_{current_number:04d}已存在，跳过")
            current_number += 1
            continue

        pair_dir.mkdir(exist_ok=True)
        shutil.copy2(img1, pair_dir / "image_01.jpg")
        shutil.copy2(img2, pair_dir / "image_02.jpg")
        print(f"✅ 保存第{current_number:04d}对: {pair_dir}")
        saved_pairs += 1
        current_number += 1

    return saved_pairs


if __name__ == "__main__":
    args = parse_arguments()
    random.seed(RANDOM_SEED)

    # 1. 加载有效图片列表（带缓存优化）
    print(f"\n=== 加载ID范围 {args.batch_start}-{args.batch_end} 内的图片 ===")
    valid_images = load_image_with_pairid(args)
    print(f"找到 {len(valid_images)} 张有效图片")

    # 2. 随机采样满足条件的配对（避免全量遍历）
    print(f"\n=== 随机抽取{args.pairs_to_extract}对（ID差≥{MIN_ID_DIFF}且pair_id不同） ===")
    try:
        random_pairs = random_sample_pairs(
            valid_images,
            args.pairs_to_extract,
            min_id_diff=MIN_ID_DIFF
        )
    except Exception as e:
        print(f"❌ 采样失败: {str(e)}")
        exit(1)

    print(f"成功获取{len(random_pairs)}对候选样本")

    # 3. 动态确定起始编号并保存
    output_dir = Path(OUTPUT_BASE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    start_number = get_next_pair_number(output_dir)
    print(f"当前已有{start_number-1}对，本次从第{start_number:04d}对开始新增")

    # 4. 保存配对（含去重和早停）
    extracted_count = save_pairs_to_output(random_pairs, output_dir, start_number)
    if extracted_count == 0:
        print("⚠️ 未成功保存任何对（可能因重复或目标数为0）")
        exit(1)

    print(f"\n🎉 处理完成！共保存{extracted_count}对中立样本，当前总对数：{start_number + extracted_count - 1}")
    print(f"所有对保存在：{output_dir}")
