import sys
import os
from pathlib import Path
from PIL import Image
import numpy as np
import logging
import traceback

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('batch_upload.log')
    ]
)
logger = logging.getLogger(__name__)

# Set up Python path
root_dir = Path(__file__).parent.resolve()
while not (root_dir / "setup.py").exists() and not (root_dir / ".git").exists():
    root_dir = root_dir.parent

groundingdino_path = root_dir / 'GroundingDINO'
sys.path.append(str(groundingdino_path))

# Import related modules
from src.core.groundingdino_handler import ClothingDetector
from src.db.uploads.image_upload import ImageUploader
# Import VectorIndex for rebuilding after batch upload
from src.core.vector_index import VectorIndex


def process_single_image(image_path):
    """Process a single image and upload segmented parts"""
    logger.info(f"开始处理图片: {image_path}")

    # Get image name (without extension)
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # Check if image file exists
    if not image_path.exists():
        logger.warning(f"图片文件 '{image_path}' 不存在，跳过处理。")
        return []

    try:
        # Open image and convert to numpy array
        logger.info(f"正在打开图片: {image_path}")
        try:
            # 使用PIL打开并确保格式正确
            image = Image.open(image_path)
            
            # 记录原始图像信息
            logger.info(f"图像格式: {image.format}, 模式: {image.mode}, 尺寸: {image.size}")
            
            # 确保图像是RGB模式
            if image.mode != 'RGB':
                logger.info(f"将图像从 {image.mode} 转换为 RGB 模式")
                image = image.convert('RGB')
                
            # 转为numpy数组
            image_np = np.array(image)
            logger.info(f"图像已转换为numpy数组, 形状: {image_np.shape}, 类型: {image_np.dtype}")
            
        except Exception as e:
            logger.error(f"无法打开或处理图片 '{image_path}': {e}")
            logger.debug(traceback.format_exc())
            return []

        # Initialize detector and detect clothes
        logger.info(f"正在检测图片中的服装: {image_path}")
        detector = ClothingDetector()
        result_bboxes = detector.detect_clothes(image_np)

        # 检查检测结果
        if not result_bboxes or len(result_bboxes) == 0:
            logger.warning(f"图片 '{image_path}' 中未检测到任何服装区域，跳过处理。")
            return []
        
        logger.info(f"在图片 '{image_path}' 中检测到 {len(result_bboxes)} 个服装区域。")

        # Create data directory
        data_dir = root_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectory for original image
        image_dir = data_dir / image_name
        image_dir.mkdir(parents=True, exist_ok=True)

        # Initialize uploader (now using fine-tuned model)
        uploader = ImageUploader()

        # Process and upload each segmented image
        uploaded_paths = []
        for idx, img_array in enumerate(result_bboxes):
            filename = f"segment_{idx}.png"
            save_path = image_dir / filename
            
            logger.info(f"正在处理和上传分割图像 {idx+1}/{len(result_bboxes)} 来自 '{image_path}'")
            
            # 检查图像数组
            if img_array is None or img_array.size == 0:
                logger.warning(f"分割区域 {idx+1} 为空，跳过处理。")
                continue
                
            # 检查图像维度和通道
            logger.debug(f"分割区域 {idx+1} 形状: {img_array.shape}, 类型: {img_array.dtype}")
            
            # 检查图像区域大小是否合理
            if img_array.shape[0] < 30 or img_array.shape[1] < 30:
                logger.warning(f"分割图像 {filename} 太小 ({img_array.shape[0]}x{img_array.shape[1]}), 跳过处理。")
                continue
                
            # 检查是否含有足够的信息（非单一颜色）
            if np.std(img_array) < 10:
                logger.warning(f"分割图像 {filename} 信息量太少（可能是单一颜色），跳过处理。")
                continue
                
            # 确保图像数据类型正确
            if img_array.dtype != np.uint8:
                logger.warning(f"图像数据类型非标准: {img_array.dtype}，转换为uint8")
                if np.issubdtype(img_array.dtype, np.floating):
                    img_array = (img_array * 255).astype(np.uint8)
                else:
                    img_array = img_array.astype(np.uint8)

            # Process and upload image
            try:
                processed_path = uploader.process_and_upload_image(img_array, idx, save_path, image_name)
                if processed_path:
                    uploaded_paths.append(processed_path)
                    logger.info(f"成功上传分割图像: {processed_path}")
                else:
                    logger.warning(f"分割图像 {filename} 上传失败，可能在处理过程中被丢弃。")
            except Exception as e:
                logger.error(f"处理分割图像 {filename} 时出错: {e}")
                logger.debug(traceback.format_exc())

        if not uploaded_paths:
            logger.warning(f"图片 '{image_path}' 没有生成任何有效的分割图像。")
        else:
            logger.info(f"图片 '{image_path}' 成功处理，生成了 {len(uploaded_paths)}/{len(result_bboxes)} 个分割图像。")
            
        return uploaded_paths

    except Exception as e:
        logger.error(f"处理图片 '{image_path}' 时出错: {e}")
        logger.debug(traceback.format_exc())
        return []


def batch_process_screenshots():
    """Batch process all images in the ScreenShots directory"""
    # Get root directory
    root_dir = Path(__file__).parent.resolve()
    while not (root_dir / "setup.py").exists() and not (root_dir / ".git").exists():
        root_dir = root_dir.parent

    # Check if ScreenShots folder exists
    screenshots_dir = root_dir / "ScreenShots"
    if not screenshots_dir.exists() or not screenshots_dir.is_dir():
        logger.error(f"'{screenshots_dir}' 文件夹不存在。请下载资源文件。")
        raise FileNotFoundError(f"'{screenshots_dir}' folder does not exist. Please download resource files.")

    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    image_files = []

    for ext in image_extensions:
        found_files = list(screenshots_dir.glob(f'*{ext}'))
        upper_files = list(screenshots_dir.glob(f'*{ext.upper()}'))
        if found_files:
            logger.info(f"找到 {len(found_files)} 个 {ext} 格式的文件")
        if upper_files:
            logger.info(f"找到 {len(upper_files)} 个 {ext.upper()} 格式的文件")
        image_files.extend(found_files)
        image_files.extend(upper_files)  # Check uppercase extensions too

    if not image_files:
        logger.warning(f"在 '{screenshots_dir}' 目录中没有找到图片文件。")
        return

    logger.info(f"找到 {len(image_files)} 个要处理的图片文件。")
    
    # 显示找到的PNG文件
    png_files = [f for f in image_files if f.suffix.lower() == '.png']
    if png_files:
        logger.info(f"其中包含 {len(png_files)} 个PNG文件:")
        for png_file in png_files:
            logger.info(f" - {png_file}")

    # Process all images
    all_uploaded_paths = []
    success_count = 0
    skip_count = 0
    
    for i, image_path in enumerate(image_files):
        logger.info(f"正在处理图片 {i+1}/{len(image_files)}: {image_path}")
        uploaded_paths = process_single_image(image_path)
        
        if uploaded_paths:
            all_uploaded_paths.extend(uploaded_paths)
            success_count += 1
        else:
            skip_count += 1
            logger.warning(f"图片 '{image_path}' 处理失败或被跳过。")

    logger.info(f"\n批处理完成! 总共处理 {len(image_files)} 个图片，成功 {success_count} 个，失败或跳过 {skip_count} 个。")
    logger.info(f"上传了 {len(all_uploaded_paths)} 个分割图像到数据库。")
    
    # Rebuild vector index after batch upload
    try:
        logger.info("正在重建向量索引...")
        vector_index = VectorIndex()
        index_result = vector_index.rebuild_index()
        if index_result[0] is not None:
            logger.info(f"向量索引重建成功，包含 {len(index_result[1])} 个向量。")
        else:
            logger.warning("警告: 向量索引重建失败，但图像已上传。")
    except Exception as e:
        logger.error(f"重建向量索引时出错: {e}")
        logger.debug(traceback.format_exc())
    
    return all_uploaded_paths


if __name__ == "__main__":
    try:
        logger.info("============== 开始批量处理图片 ==============")
        # Execute batch processing
        uploaded_files = batch_process_screenshots()

        # Print processing results
        if uploaded_files:
            logger.info("\n成功上传文件到 split_images 表:")
            for path in uploaded_files:
                logger.info(f" - {path}")
            
            logger.info("\n图像已使用微调的 DINOv2 模型处理。")
        else:
            logger.warning("\n没有成功上传任何文件。")

    except Exception as e:
        logger.error(f"批处理过程中出错: {e}")
        logger.error(traceback.format_exc())
    
    logger.info("============== 批量处理图片结束 ==============")