import tempfile

import numpy as np
from PIL import Image
from pathlib import Path
# from image_splicing_detection import ImageSplicingDetector

INPUT_DIR = Path("/Users/sunyuliang/Desktop/AppBuilder/Python/ScreenShots")  # 截图输入目录
OUTPUT_DIR = Path("/Users/sunyuliang/Desktop/AppBuilder/Python/images_upload")  # 处理结果目录
BOX_THRESHOLD = 0.5  # GroundingDINO检测阈值
TEXT_PROMPT = "clothes, garment"  # 检测目标文本提示
SIMILARITY_THRESHOLD = 0.4  # 与基准特征的相似度阈值


def process_screenshots():
    """处理截图目录的主函数"""
    from src.core.groundingdino_handler import ClothingDetector
    from src.utils.image_judge import ImageJudge
    import traceback
    
    # 新增检测器实例化
    # splicing_detector = ImageSplicingDetector()  # 新增行
    
    detector = ClothingDetector(box_threshold=0.25)
    judger = ImageJudge()
    
    # 添加目录验证
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"输入目录不存在: {INPUT_DIR}")
    print(f"输入目录检测到 {len(list(INPUT_DIR.glob('*.*')))} 个文件")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    processed_count = 0

    # 修改为支持多种图片格式
    for img_path in INPUT_DIR.glob("*.*"):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
            try:
                print(f"\n开始处理 {img_path.name}")  # 新增进度提示
                image = Image.open(img_path).convert("RGB")
                segments = detector.detect_clothes(np.array(image))
                
                if not segments:
                    continue

                # 判断并保存有效片段
                segment_counter = 0  # 独立计数器
                for idx, segment in enumerate(segments):
                    segment_img = Image.fromarray(segment).convert('RGB')
                    
                    # 使用微调模型进行判断
                    is_cloth, score = judger.is_clothing(segment_img)
                    
                    if is_cloth:
                        # 修改后的检测调用
                        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
                            segment_img.save(tmp.name)
                            # if splicing_detector.detect_image_splicing(tmp.name):  # 修改调用方式
                            #     print(f"⚠️ 检测到拼接痕迹，跳过保存: {img_path.stem}_segment_{segment_counter}")
                            #     continue
                        
                        filename = f"{img_path.stem}_segment_{segment_counter}.jpg"
                        save_path = OUTPUT_DIR / filename
                        segment_img.save(save_path)
                        processed_count += 1
                        segment_counter += 1
                        print(f"✅ 保存有效片段: {filename} 相似度: {score:.4f}")

            except Exception as e:
                print(f"处理失败 {img_path.name}:")
                print(traceback.format_exc())  # 打印完整错误堆栈

    print(f"\n处理完成，共保存{processed_count}个有效衣物片段")


if __name__ == "__main__":
    process_screenshots()
