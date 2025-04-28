from .image_similarity_DINOv2_finetuned import ImageSimilarityDINOv2Finetuned
from pathlib import Path

# 获取项目根目录
root_dir = Path(__file__).parent.parent.parent.parent

# 使用准确的模型路径
model_path = root_dir / "src" / "training" / "models" / "best_model.pth"

# 检查模型文件是否存在
if not model_path.exists():
    raise FileNotFoundError(f"找不到模型文件: {model_path}，请确认模型文件位置")
else:
    print(f"找到模型文件: {model_path}")

# 创建实例
ImageSimilarity = ImageSimilarityDINOv2Finetuned(model_path=str(model_path))
