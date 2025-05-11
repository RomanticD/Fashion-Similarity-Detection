from pathlib import Path
from PIL import Image


class ImageJudge:
    def __init__(self):
        # 修改为基于工作区目录的路径配置
        self.root_dir = Path("/Users/sunyuliang/Desktop/AppBuilder/Python/RD-Test")  # 项目根目录
        self.model_path = self.root_dir / "src/training/models/best_model.pth"  # 相对项目根目录的路径
        self.base_image_path = self.root_dir / "Assets/base_clothes.jpg"
        
        # 添加路径验证
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件未找到: {self.model_path}")
        if not self.base_image_path.exists():
            raise FileNotFoundError(f"基准图像未找到: {self.base_image_path}")
            
        # 初始化模型和基准特征
        self.model = None
        self.base_feature = None
        self._initialize_model()
        self._load_base_feature()

    def _initialize_model(self):
        """私有方法初始化模型"""
        if self.model is None:
            try:
                from src.core.image_similarity.image_similarity_DINOv2_finetuned import ImageSimilarityDINOv2Finetuned
                self.model = ImageSimilarityDINOv2Finetuned(model_path=str(self.model_path))
                print(f"✅ 微调模型加载成功: {self.model_path}")
            except Exception as e:
                print(f"❗ 模型初始化失败: {str(e)}")
                raise

    def _load_base_feature(self):
        """私有方法加载基准特征"""
        if self.base_feature is None:
            try:
                img = Image.open(self.base_image_path).convert('RGB')
                self.base_feature = self.model.extract_feature(img)
                print(f"✅ 基准特征提取完成，维度: {len(self.base_feature)}")
            except Exception as e:
                print(f"❗ 基准特征提取失败: {str(e)}")
                raise

    def is_clothing(self, image_input, threshold=0.4):
        """公开判断方法"""
        try:
            # 处理输入类型
            if isinstance(image_input, (str, Path)):
                img = Image.open(image_input).convert('RGB')
            elif isinstance(image_input, Image.Image):
                img = image_input.convert('RGB')
            else:
                raise ValueError("不支持的输入类型")
            
            # 特征提取
            current_feature = self.model.extract_feature(img)
            if current_feature is None or len(current_feature) == 0:
                raise ValueError("当前特征提取失败")

            # 计算相似度
            similarity = self.model.cosine_similarity(self.base_feature, current_feature)
            return similarity >= threshold, round(similarity, 4)

        except Exception as e:
            print(f"检测失败: {str(e)}")
            return False, 0.0


if __name__ == "__main__":
    # 使用示例
    judger = ImageJudge()
    test_image = "/Users/sunyuliang/Desktop/AppBuilder/Python/RD-Test/data/test_groundingDINO/test_06/segment_5.png"
    result, score = judger.is_clothing(test_image)
    print(f"检测结果: {'衣物' if result else '非衣物'}, 相似度: {score}")
