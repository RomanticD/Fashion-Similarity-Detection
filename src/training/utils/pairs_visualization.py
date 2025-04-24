# Fashion-Similarity-Detection/training/utils/pairs_visualization.py
import shutil
from pathlib import Path

# ======================
# 可调整参数（集中在开头）
# ======================
INPUT_PAIRS_DIR = Path("/Users/sunyuliang/Desktop/AppBuilder/Python/dinov2_train/training_pairs")  # 原始对文件夹路径
OUTPUT_EXAMINE_DIR = Path("/Users/sunyuliang/Desktop/AppBuilder/Python/examine")  # 可视化目标目录
START_PAIR = 1201  # 起始对号（如1对应pair_0001）
END_PAIR = 1250  # 结束对号（如100对应pair_0100）


def main():
    # 创建目标目录
    OUTPUT_EXAMINE_DIR.mkdir(parents=True, exist_ok=True)
    missing_pairs = []  # 记录缺失的对号

    for pair_id in range(START_PAIR, END_PAIR + 1):
        pair_dir = INPUT_PAIRS_DIR / f"pair_{pair_id:04d}"
        if not pair_dir.exists():
            missing_pairs.append(pair_id)
            continue  # 跳过不存在的对

        # 处理该对中的两张图片
        for img_idx, img_type in enumerate(["image_01", "image_02"], start=1):
            img_path = pair_dir / f"{img_type}.jpg"
            if not img_path.exists():
                continue  # 跳过不存在的图片

            # 生成目标文件名：pair_0001_1.jpg 和 pair_0001_2.jpg
            output_name = f"pair_{pair_id:04d}_{img_idx}.jpg"
            output_path = OUTPUT_EXAMINE_DIR / output_name

            # 复制文件
            shutil.copy2(img_path, output_path)
            print(f"✅ 复制 {img_path.name} 到 {output_path.name}")

    # 输出缺失对号列表
    if missing_pairs:
        print(f"\n⚠️ 以下对号不存在或路径错误: {missing_pairs}")
    else:
        print("\n✅ 所有指定对号均已处理，无缺失项")

    print(f"\n📁 可视化结果已保存至: {OUTPUT_EXAMINE_DIR}")


if __name__ == "__main__":
    main()
