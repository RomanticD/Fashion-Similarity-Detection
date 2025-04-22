# src/training/data/create_pairs.py
import os
from pathlib import Path
from PIL import Image  # 新增PIL检查


def is_image_file(file_path):
    """检查文件是否为支持的图像格式（JPEG/PNG/BMP等）"""
    try:
        with Image.open(file_path) as img:
            return img.format in ['JPEG', 'PNG', 'BMP', 'GIF', 'TIFF']  # 支持的格式列表
    except Exception:
        return False


def generate_pair_list(data_dir, output_file):
    with open(output_file, 'w') as f:
        # 正样本对处理
        for pair_type in ['positive_pairs', 'negative_pairs']:
            pair_dir = data_dir / pair_type
            if not pair_dir.exists():
                continue
            for sub_dir in pair_dir.glob("pair_*"):
                if not sub_dir.is_dir():
                    continue
                # 收集所有图像文件（不依赖扩展名）
                imgs = [f for f in sub_dir.glob("*") if is_image_file(f)]
                if len(imgs) == 2:
                    label = 1 if "positive" in pair_type else 0
                    f.write(f"{imgs[0]},{imgs[1]},{label}\n")


if __name__ == "__main__":
    data_dir = Path("data/finetune_data")
    generate_pair_list(data_dir, data_dir / "train.txt")
    generate_pair_list(data_dir, data_dir / "val.txt")
