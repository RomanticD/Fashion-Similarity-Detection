import random
import shutil
import json
from pathlib import Path
from PIL import Image
import numpy as np
from src.core.groundingdino_handler import ClothingDetector  # 项目内置GroundingDINO处理器

# ======================
# 可调整参数（集中在开头，支持结构化路径）
# ======================
DEEPFASHION_BASE_DIR = Path("/Users/sunyuliang/Desktop/AppBuilder/Python/deepfashion_train")  # DeepFashion2根目录
IMAGE_SUB_DIR = "image"               # 图片子文件夹名
ANNO_SUB_DIR = "annos"               # 标注子文件夹名
PASS_DATASET_DIR = Path("/Users/sunyuliang/Desktop/AppBuilder/Python/PASS_dataset")  # PASS数据集路径
OUTPUT_PAIRS_DIR = Path("/Users/sunyuliang/Desktop/AppBuilder/Python/dinov2_train/training_pairs")  # 输出对目录
TEXT_PROMPT = "clothes, garment, clothing item"         # GroundingDINO检测提示词
BOX_THRESHOLD = 0.3                                     # 检测阈值（0-1）
FORCE_CROP = True                                       # 未检测到衣物时是否强制使用原图
RANDOM_SEED = 42                                        # 随机种子（固定以确保可复现性）
NUM_NEG_PAIRS = 215                                      # 生成的负样本对数
MIN_ID_RANGE = 50001                                   # 最小图片ID（前6位）
MAX_ID_RANGE = 100000                                   # 最大图片ID（前6位）


# ======================
# 核心函数定义
# ======================
def image_id_from_filename(filename):
    """从文件名中提取6位数值ID（如"050001.jpg" → 50001）"""
    stem = Path(filename).stem
    return int(stem[:6]) if len(stem) >= 6 else None


def load_valid_clothing_images(base_dir, image_dir=IMAGE_SUB_DIR, anno_dir=ANNO_SUB_DIR):
    """加载有效衣物图片（含标注且有pair_id）"""
    image_dir_path = base_dir / image_dir
    anno_dir_path = base_dir / anno_dir
    valid_images = []
    anno_cache = {}  # 缓存标注文件存在性

    # 预检查标注文件存在性
    for img_id in range(MIN_ID_RANGE, MAX_ID_RANGE + 1):
        anno_path = anno_dir_path / f"{img_id:06d}.json"
        anno_cache[img_id] = anno_path.exists()

    # 遍历图片文件并筛选有效项
    for img_file in image_dir_path.glob("*.jpg"):
        try:
            img_id = image_id_from_filename(img_file)
            if not img_id or not (MIN_ID_RANGE <= img_id <= MAX_ID_RANGE):
                continue  # 跳过无效ID或范围外的图片

            if not anno_cache.get(img_id, False):
                print(f"警告：跳过无标注文件的图片 {img_file.name}")
                continue

            # 读取pair_id
            with open(anno_dir_path / f"{img_id:06d}.json", "r") as f:
                anno_data = json.load(f)
                pair_id = anno_data.get("pair_id")

            if not pair_id:
                print(f"警告：跳过无pair_id的图片 {img_file.name}")
                continue

            valid_images.append(str(img_file))  # 保存图片路径字符串
        except Exception as e:
            print(f"处理图片 {img_file.name} 失败: {str(e)}")
            continue

    if not valid_images:
        raise ValueError(f"在ID范围 {MIN_ID_RANGE}-{MAX_ID_RANGE} 内未找到有效衣物图片")
    return valid_images


def get_random_clothing_image(valid_image_list):
    """从有效图片列表中随机获取一张图片路径"""
    return random.choice(valid_image_list)


def process_with_groundingdino(image_path, detector, force_crop=True):
    """使用GroundingDINO处理图片并返回裁剪后的图像（带异常处理）"""
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
                print(f"警告: 未检测到衣物，使用原图处理: {Path(image_path).name}")
                return image  # 返回原图
            else:
                print(f"跳过: 未检测到衣物: {Path(image_path).name}")
                return None

        return Image.fromarray(segmented_images[0])  # 取第一个检测区域
    except Exception as e:
        print(f"处理失败: {Path(image_path).name}, 错误: {str(e)}")
        return None


def get_random_non_clothing_image(pass_dir):
    """从PASS数据集递归获取随机非衣物图片（处理乱码文件名）"""
    all_images = list(pass_dir.glob("**/*.jpg")) + list(pass_dir.glob("**/*.JPG"))
    if not all_images:
        raise ValueError("PASS数据集下未找到有效图片")
    return random.choice(all_images)


def create_negative_pair(detector, output_dir, pair_number, valid_cloth_list):
    """创建单个负样本对（去除权限相关操作）"""
    pair_dir = output_dir / f"pair_{pair_number:04d}"
    if pair_dir.exists():
        print(f"警告: pair_{pair_number:04d} 已存在，跳过")
        return False

    # 1. 随机获取有效衣物图片并处理
    cloth_img_path = get_random_clothing_image(valid_cloth_list)
    processed_cloth = process_with_groundingdino(cloth_img_path, detector, FORCE_CROP)
    if processed_cloth is None:
        return False  # 处理失败则跳过该对

    # 2. 获取非衣物图片
    non_cloth_img_path = get_random_non_clothing_image(PASS_DATASET_DIR)

    # 3. 保存配对
    pair_dir.mkdir(exist_ok=True)
    processed_cloth.save(pair_dir / "image_01.jpg")
    shutil.copy2(non_cloth_img_path, pair_dir / "image_02.jpg")
    print(f"✅ 生成第{pair_number:04d}对: {pair_dir}")
    return True


# ======================
# 主流程（去除权限相关逻辑）
# ======================
if __name__ == "__main__":
    detector = ClothingDetector()
    detector.box_threshold = BOX_THRESHOLD

    # 1. 加载有效衣物图片列表
    print(f"\n=== 加载ID范围 {MIN_ID_RANGE}-{MAX_ID_RANGE} 内的有效衣物图片 ===")
    valid_cloth_images = load_valid_clothing_images(DEEPFASHION_BASE_DIR)
    print(f"找到 {len(valid_cloth_images)} 张有效衣物图片")

    # 2. 初始化输出目录
    OUTPUT_PAIRS_DIR.mkdir(parents=True, exist_ok=True)

    # 3. 计算当前最大对号
    existing_pairs = [d.name for d in OUTPUT_PAIRS_DIR.glob("pair_????") if d.is_dir()]
    start_pair = 1 if not existing_pairs else max(int(p.split('_')[1]) for p in existing_pairs) + 1

    successful_pairs = 0
    current_pair = start_pair

    # 4. 生成负样本对（带重试机制）
    print(f"\n=== 开始生成{NUM_NEG_PAIRS}对负样本（从第{start_pair:04d}对开始） ===")
    while successful_pairs < NUM_NEG_PAIRS and current_pair < start_pair + 2 * NUM_NEG_PAIRS:
        if create_negative_pair(detector, OUTPUT_PAIRS_DIR, current_pair, valid_cloth_images):
            successful_pairs += 1
        current_pair += 1

    print(f"\n🎉 完成！共成功生成{successful_pairs}对负样本")
    print(f"所有对保存在: {OUTPUT_PAIRS_DIR}")
