# Fashion-Similarity-Detection/training/utils/positive_pairs_extract.py
import json
import shutil
from pathlib import Path
import argparse
import random
from collections import defaultdict

# ======================
# 可调整参数（集中在开头，支持命令行覆盖）
# ======================
INPUT_DIR = "/Users/sunyuliang/Desktop/AppBuilder/Python/deepfashion_train"  # 原始数据目录
IMAGE_DIR_NAME = "image"         # 图片文件夹名（如"image"或"images"）
ANNO_DIR_NAME = "annos"          # 标注文件夹名
OUTPUT_BASE_DIR = "/Users/sunyuliang/Desktop/AppBuilder/Python/dinov2_train/training_pairs"  # 输出基目录
RANDOM_SEED = 42                 # 随机种子（固定以确保可复现性）


def parse_arguments():
    parser = argparse.ArgumentParser(description='随机抽取正样本对（支持6位文件名/动态序号）')
    parser.add_argument('--batch_start', type=int, default=20001,
                        help='批次起始数值ID（如1对应000001.json）')
    parser.add_argument('--batch_end', type=int, default=30000,
                        help='批次结束数值ID（如10000对应010000.json）')
    parser.add_argument('--pairs_to_extract', type=int, default=500,
                        help='本次提取的正样本对数')
    parser.add_argument('--max_id_diff', type=int, default=1,
                        help='同款图片ID最大间隔（如1表示仅相邻ID配对）')
    return parser.parse_args()


def image_id_from_filename(filename):
    """从6位文件名提取数值ID（如"010000.json"→10000）"""
    return int(filename.stem)  # 直接获取文件名前缀（不含.json后缀）


def load_batch_pairs(args):
    """按数值ID排序加载批次数据，生成6位图片路径"""
    input_path = Path(INPUT_DIR)
    anno_dir = input_path / ANNO_DIR_NAME
    image_dir = input_path / IMAGE_DIR_NAME

    # 按数值ID排序标注文件（关键修正：处理6位文件名）
    all_anno_files = sorted(
        anno_dir.glob("*.json"),
        key=lambda f: image_id_from_filename(f)  # 按数值ID升序排列
    )
    batch_files = [
        f for f in all_anno_files if
        args.batch_start <= image_id_from_filename(f) <= args.batch_end
    ]

    pair_group = defaultdict(list)
    for anno_file in batch_files:
        img_id = image_id_from_filename(anno_file)
        # 生成6位图片文件名（如1→000001.jpg）
        img_path = None
        for suffix in [".jpg", ".jpeg", ".png"]:
            candidate = image_dir / f"{img_id:06d}{suffix}"  # 补全6位文件名
            if candidate.exists():
                img_path = str(candidate)
                break
        if not img_path:
            continue

        try:
            with open(anno_file, 'r') as f:
                data = json.load(f)
            pair_id = data.get("pair_id")
            if pair_id:
                pair_group[pair_id].append( (img_id, img_path) )
        except Exception as e:
            print(f"跳过损坏的JSON文件{anno_file.name}: {str(e)}")
            continue

    return pair_group


def filter_adjacent_pairs(pair_group, max_id_diff):
    """提取所有符合条件的相邻ID对，并随机排序"""
    valid_pairs = []
    for img_list in pair_group.values():
        sorted_imgs = sorted(img_list, key=lambda x: x[0])  # 按ID排序
        for i in range(len(sorted_imgs) - 1):
            current_id, current_path = sorted_imgs[i]
            next_id, next_path = sorted_imgs[i+1]
            if next_id - current_id <= max_id_diff:
                valid_pairs.append( (current_path, next_path) )  # 保存图片路径对

    # 随机打乱有效对（确保每次抽取不同组合）
    random.seed(RANDOM_SEED)
    random.shuffle(valid_pairs)
    return valid_pairs


def get_next_pair_number(output_dir):
    """获取当前输出目录的最大对编号，支持不连续编号"""
    existing_pairs = [d.name for d in Path(output_dir).glob("pair_????") if d.is_dir()]
    if not existing_pairs:
        return 1
    # 提取4位数字编号（如pair_0012→12）
    numbers = [int(d.split('_')[1]) for d in existing_pairs]
    return max(numbers) + 1


def save_pairs_to_output(pairs, output_dir, start_number):
    """按动态序号保存配对，延续人工筛选后的编号"""
    output_path = Path(output_dir)
    current_number = start_number
    saved_pairs = 0

    for img1, img2 in pairs:
        if saved_pairs >= args.pairs_to_extract:  # 达到目标数量后停止
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

    print(f"本次新增{saved_pairs}对，从第{start_number:04d}对开始")
    return saved_pairs


if __name__ == "__main__":
    args = parse_arguments()
    random.seed(RANDOM_SEED)

    # 1. 加载批次数据（处理6位文件名）
    print(f"\n=== 处理批次 {args.batch_start}-{args.batch_end} ===")
    pair_group = load_batch_pairs(args)
    if not pair_group:
        print("❌ 该批次无有效标注文件或图片（检查6位文件名是否正确）")
        exit(1)

    # 2. 提取所有符合条件的相邻对并随机化
    valid_adjacent_pairs = filter_adjacent_pairs(pair_group, args.max_id_diff)
    if not valid_adjacent_pairs:
        print(f"❌ 未找到相邻ID≤{args.max_id_diff}的配对（检查数据分布）")
        exit(1)

    # 3. 随机抽取目标数量的对（允许超过后截断）
    random_pairs = random.sample(valid_adjacent_pairs, min(args.pairs_to_extract, len(valid_adjacent_pairs)))
    print(f"从{len(valid_adjacent_pairs)}个候选对中随机抽取{len(random_pairs)}对")

    # 4. 动态确定起始编号并保存
    output_dir = Path(OUTPUT_BASE_DIR)
    start_number = get_next_pair_number(output_dir)
    print(f"当前已有{start_number-1}对，本次从第{start_number:04d}对开始新增")

    # 5. 保存配对（含去重和早停）
    extracted_count = save_pairs_to_output(random_pairs, output_dir, start_number)
    if extracted_count == 0:
        print("⚠️ 未成功保存任何对（可能因重复或目标数为0）")
        exit(1)

    print(f"\n🎉 处理完成！共保存{extracted_count}对，当前总对数：{start_number + extracted_count - 1}")
    print(f"所有对保存在：{output_dir}")
