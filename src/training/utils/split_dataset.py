import os
import random
from pathlib import Path


def split_data(data_dir, train_ratio=0.8):
    positive_dir = data_dir / "positive_pairs"
    negative_dir = data_dir / "negative_pairs"
    all_pairs = []

    # 收集所有正样本对
    for pair_dir in positive_dir.glob("pair_*"):
        all_pairs.append((str(pair_dir), 1))

    # 收集所有负样本对
    for pair_dir in negative_dir.glob("pair_*"):
        all_pairs.append((str(pair_dir), 0))

    # 随机打乱并划分
    random.shuffle(all_pairs)
    split_idx = int(len(all_pairs) * train_ratio)
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]

    # 生成数据列表
    def write_list(pairs, output_file):
        with open(output_file, 'w') as f:
            for pair_dir, label in pairs:
                img1 = os.path.join(pair_dir, os.listdir(pair_dir)[0])
                img2 = os.path.join(pair_dir, os.listdir(pair_dir)[1])
                f.write(f"{img1},{img2},{label}\n")

    write_list(train_pairs, data_dir / "train.txt")
    write_list(val_pairs, data_dir / "val.txt")


if __name__ == "__main__":
    data_dir = Path("data/finetune_data")
    split_data(data_dir)
