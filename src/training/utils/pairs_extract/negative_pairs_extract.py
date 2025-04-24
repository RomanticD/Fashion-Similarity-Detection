# Fashion-Similarity-Detection/training/utils/negative_pairs_extract.py
import random
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from src.core.groundingdino_handler import ClothingDetector  # 项目内置GroundingDINO处理器

# ======================
# 可调整参数（集中在开头）
# ======================
DEEPFASHION_DIR = Path("/Users/sunyuliang/Desktop/AppBuilder/Python/deepfashion_train")  # DeepFashion2原始衣物图片目录
PASS_DATASET_DIR = Path("/Users/sunyuliang/Desktop/AppBuilder/Python/PASS_dataset")  # PASS数据集路径（含0-19子文件夹）
OUTPUT_PAIRS_DIR = Path("/Users/sunyuliang/Desktop/AppBuilder/Python/dinov2_train/training_pairs")  # 输出对目录
TEXT_PROMPT = "clothes, garment, clothing item"         # GroundingDINO检测提示词
BOX_THRESHOLD = 0.3                                     # 检测阈值（0-1）
FORCE_CROP = True                                       # 未检测到衣物时是否强制使用原图
RANDOM_SEED = 42                                        # 随机种子（固定以确保可复现性）
NUM_NEG_PAIRS = 250                                    # 生成的负样本对数


# ======================
# 核心函数定义
# ======================
def get_random_clothing_image(deepfashion_dir):
    """从DeepFashion2中随机获取一张有效衣物图片路径"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG']
    all_images = [f for f in deepfashion_dir.glob('*') if f.suffix.lower() in image_extensions]
    if not all_images:
        raise ValueError("DeepFashion2目录下未找到有效图片")
    random.seed(RANDOM_SEED)
    return random.choice(all_images)


def process_with_groundingdino(image_path, detector, force_crop=True):
    """使用GroundingDINO处理图片并返回裁剪后的图像"""
    try:
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        segmented_images = detector.detect_clothes(
            image_np,
            text_prompt=TEXT_PROMPT,
            box_threshold=BOX_THRESHOLD
        )

        if not segmented_images:
            if force_crop:
                print(f"警告: 未检测到衣物，使用原图处理: {image_path.name}")
                return image  # 返回原图
            else:
                print(f"跳过: 未检测到衣物: {image_path.name}")
                return None

        return Image.fromarray(segmented_images[0])  # 取第一个检测到的区域（假设单衣物）

    except Exception as e:
        print(f"处理失败: {image_path.name}, 错误: {str(e)}")
        return None


def get_random_non_clothing_image(pass_dir):
    """从PASS数据集的所有子文件夹中随机获取一张非衣物图片路径"""
    # 递归获取所有子文件夹中的JPG文件（忽略文件夹名称，处理乱码文件名）
    all_images = list(pass_dir.glob("**/*.jpg")) + list(pass_dir.glob("**/*.JPG"))
    if not all_images:
        raise ValueError("PASS数据集下未找到有效图片")
    random.seed(RANDOM_SEED)
    return random.choice(all_images)


def create_negative_pair(detector, output_dir, pair_number):
    """创建单个负样本对（衣物+非衣物）"""
    pair_dir = output_dir / f"pair_{pair_number:04d}"
    if pair_dir.exists():
        print(f"警告: pair_{pair_number:04d} 已存在，跳过")
        return False

    # 1. 处理衣物图片
    cloth_img_path = get_random_clothing_image(DEEPFASHION_DIR)
    processed_cloth = process_with_groundingdino(cloth_img_path, detector, FORCE_CROP)
    if processed_cloth is None:
        return False  # 处理失败则跳过该对

    # 2. 获取非衣物图片（从PASS的子文件夹中随机选取）
    non_cloth_img_path = get_random_non_clothing_image(PASS_DATASET_DIR)

    # 3. 保存配对
    pair_dir.mkdir(exist_ok=True)
    processed_cloth.save(pair_dir / "image_01.jpg")
    shutil.copy2(non_cloth_img_path, pair_dir / "image_02.jpg")
    print(f"✅ 生成第{pair_number:04d}对: {pair_dir}")
    return True


# ======================
# 主流程
# ======================
if __name__ == "__main__":
    detector = ClothingDetector()
    detector.box_threshold = BOX_THRESHOLD

    # 初始化输出目录
    OUTPUT_PAIRS_DIR.mkdir(parents=True, exist_ok=True)

    # 计算当前最大对号
    existing_pairs = [d.name for d in OUTPUT_PAIRS_DIR.glob("pair_????") if d.is_dir()]
    start_pair = 1 if not existing_pairs else max(int(p.split('_')[1]) for p in existing_pairs) + 1

    successful_pairs = 0  # 记录成功生成的对号
    current_pair = start_pair  # 初始化当前对号

    # 生成负样本对
    while successful_pairs < NUM_NEG_PAIRS:
        if create_negative_pair(detector, OUTPUT_PAIRS_DIR, current_pair):
            successful_pairs += 1
        current_pair += 1  # 无论成功与否都递增对号，确保不重复

    print(f"\n🎉 完成！共生成{successful_pairs}对负样本，起始对号: {start_pair}，结束对号: {current_pair - 1}")
    print(f"所有对保存在: {OUTPUT_PAIRS_DIR}")
