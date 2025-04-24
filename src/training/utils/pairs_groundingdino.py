from pathlib import Path
from PIL import Image
import numpy as np

from src.core.groundingdino_handler import ClothingDetector  # 项目内置GroundingDINO处理器

# ======================
# 可调整参数（直接在代码顶部手动定义）
# ======================
INPUT_PAIRS_DIR = Path("/Users/sunyuliang/Desktop/AppBuilder/Python/dinov2_train/training_pairs")  # positive_pairs根目录
TEXT_PROMPT = "clothes, garment, clothing item"  # 检测提示词（支持多语言）
BOX_THRESHOLD = 0.3  # 检测阈值（降低至0.15提高灵敏度）
FORCE_PROCESS = True   # 未检测到服装时是否强制使用原图

# 手动定义开始对和结束对（直接修改以下两行）
START_PAIR = 1251       # 起始pair编号（如10对应pair_0010）
END_PAIR = 1750         # 结束pair编号（如50对应pair_0050）


def get_valid_image_paths(pairs_dir, start_pair, end_pair):
    """动态获取有效pair中的图片路径（跳过缺失对），同时记录对号和图片类型"""
    image_info = []  # 存储 (pair_id, img_type, img_path) 元组
    for pair_id in range(start_pair, end_pair + 1):
        pair_dir = pairs_dir / f"pair_{pair_id:04d}"
        if not pair_dir.is_dir():
            print(f"警告: pair_{pair_id:04d} 不存在，跳过该对")
            continue  # 跳过不存在的对

        for img_type in ["image_01", "image_02"]:
            img_path = pair_dir / f"{img_type}.jpg"
            if not img_path.exists():
                print(f"警告: {img_path.name} 不存在，跳过该图片")
                continue
            image_info.append((pair_id, img_type, img_path))  # 保存详细信息
    return image_info  # 返回包含对号和图片类型的元组列表


def process_image(pair_id, img_type, image_path, detector, force_process=True, undetected_list=None):
    """处理单张图片，返回分割后的图像或原图，包含对号和图片类型信息，并记录未检测情况"""
    try:
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        segmented_images = detector.detect_clothes(image_np, text_prompt=TEXT_PROMPT, box_threshold=BOX_THRESHOLD)

        if not segmented_images:
            # 记录未检测到的图片信息
            undetected_list.append(f"第{pair_id}对的{img_type}")
            if force_process:
                print(f"警告: 第{pair_id}对的{img_type}未检测到服装，使用原图")
                return image  # 返回原图
            else:
                print(f"警告: 第{pair_id}对的{img_type}未检测到服装，跳过")
                return None

        return Image.fromarray(segmented_images[0])

    except Exception as e:
        print(f"处理第{pair_id}对的{img_type}失败: {str(e)}")
        return None


def crop_and_replace(detector, pair_id, img_type, image_path, force_process=True, undetected_list=None):
    """裁剪并替换原图，包含对号和图片类型信息，并传递未检测列表"""
    processed_img = process_image(pair_id, img_type, image_path, detector, force_process, undetected_list)
    if processed_img is None:
        return False

    try:
        processed_img.save(image_path)
        print(f"✅ 处理完成: 第{pair_id}对的{img_type}，新尺寸: {processed_img.size}")
        return True
    except Exception as e:
        print(f"保存第{pair_id}对的{img_type}失败: {str(e)}")
        return False


def main():
    detector = ClothingDetector()
    detector.box_threshold = BOX_THRESHOLD
    image_info_list = get_valid_image_paths(INPUT_PAIRS_DIR, START_PAIR, END_PAIR)

    if not image_info_list:
        print("❌ 未找到任何有效图片")
        return

    undetected_images = []  # 用于记录未检测到的图片
    print(f"开始处理 {len(image_info_list)} 张图片（从pair_{START_PAIR:04d} 到 pair_{END_PAIR:04d}）")

    for pair_id, img_type, img_path in image_info_list:
        crop_and_replace(detector, pair_id, img_type, img_path, FORCE_PROCESS, undetected_images)

    # 打印未检测到的图片汇总
    if undetected_images:
        print(f"\n⚠️ 以下图片未成功检测到服装:")
        for item in undetected_images:
            print(f"- {item}")
    else:
        print("\n✅ 所有处理的图片均成功检测到服装")

    print(f"\n🎉 处理完成！成功处理 {len(image_info_list)} 张图片")


if __name__ == "__main__":
    main()
