# Fashion-Similarity-Detection/training/utils/data_preprocess.py
import json
import shutil
from pathlib import Path
import argparse


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='从训练数据集提取正样本对（相似度1）')
    parser.add_argument('--input_dir', type=str, default='train',
                        help='输入数据集目录（包含annos.json和images文件夹）')
    parser.add_argument('--output_dir', type=str, default='data/training/positive',
                        help='正样本对输出目录')
    parser.add_argument('--num_pairs', type=int, default=10,
                        help='提取的样本对数量')
    return parser.parse_args()


def load_annotation(anno_path):
    """加载标注文件，提取正样本对路径"""
    with open(anno_path, 'r') as f:
        data = json.load(f)
    # 假设annos.json格式为：{"pairs": [[image1_path, image2_path], ...]}
    # 请根据实际格式调整，以下为示例结构
    positive_pairs = data.get('positive_pairs', [])[:args.num_pairs]
    return positive_pairs


def create_file_structure(output_dir, num_pairs):
    """创建输出目录结构：positive/pair_001, pair_002, ..."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num_pairs):
        pair_dir = output_dir / f"pair_{i+1:03d}"
        pair_dir.mkdir(exist_ok=True)
    return output_dir


def copy_image_pairs(input_image_dir, output_dir, pairs):
    """复制图片对到对应文件夹"""
    input_image_dir = Path(input_image_dir)
    for idx, (img1, img2) in enumerate(pairs, 1):
        pair_dir = output_dir / f"pair_{idx:03d}"
        # 复制第一张图片
        src_path1 = input_image_dir / img1
        dst_path1 = pair_dir / f"image_01.jpg"  # 统一命名格式
        shutil.copyfile(src_path1, dst_path1)
        # 复制第二张图片
        src_path2 = input_image_dir / img2
        dst_path2 = pair_dir / f"image_02.jpg"
        shutil.copyfile(src_path2, dst_path2)
        print(f"✅ 复制第{idx}组正样本对到 {pair_dir}")


def save_pair_metadata(output_dir, pairs):
    """（可选）保存配对元数据（类似DeepFashion的标注格式）"""
    metadata = []
    for idx, (img1, img2) in enumerate(pairs, 1):
        metadata.append({
            "pair_id": f"pair_{idx:03d}",
            "image1": f"pair_{idx:03d}/image_01.jpg",
            "image2": f"pair_{idx:03d}/image_02.jpg",
            "similarity_label": 1.0  # 正样本标签
        })
    metadata_path = Path(output_dir) / "positive_pairs_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"📄 保存配对元数据到 {metadata_path}")


if __name__ == "__main__":
    args = parse_arguments()

    # 路径定义
    input_anno_path = Path(args.input_dir) / "annos.json"
    input_image_dir = Path(args.input_dir) / "images"
    output_dir = Path(args.output_dir)

    # 1. 加载正样本对标注
    positive_pairs = load_annotation(input_anno_path)

    # 2. 创建输出目录结构
    create_file_structure(output_dir, args.num_pairs)

    # 3. 复制图片对
    copy_image_pairs(input_image_dir, output_dir, positive_pairs)

    # 4. （可选）保存配对元数据（推荐添加，兼容DeepFashion格式）
    save_pair_metadata(output_dir, positive_pairs)

    print(f"🎉 成功提取{args.num_pairs}组正样本对到 {output_dir}")
