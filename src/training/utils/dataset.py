from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from src.core.image_similarity.image_similarity_DINOv2 import ImageSimilarityDINOv2  # 导入DINOv2预处理配置


class MultiFormatPairDataset(Dataset):
    def __init__(self, list_file, image_size=224):
        """
        支持多格式图像对的数据集类
        :param list_file: 数据列表文件路径（每行：img1_path,img2_path,label）
        :param image_size: 输入图像尺寸（需与DINOv2预处理一致，默认224）
        """
        self.lines = open(list_file, 'r').readlines()
        self.image_size = image_size

        # 初始化预处理（与DINOv2完全一致）
        self.transform = transforms.Compose([
            transforms.Resize(image_size, antialias=True),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __getitem__(self, idx):
        """加载并预处理图像对，返回Tensor和标签"""
        img1_path, img2_path, label = self.lines[idx].strip().split(',')

        # 1. 多格式图像加载（支持JPEG/PNG/BMP等，自动转换为RGB）
        img1 = self._load_image(img1_path)
        img2 = self._load_image(img2_path)

        # 2. 应用预处理
        img1_tensor = self.transform(img1)
        img2_tensor = self.transform(img2)

        return (img1_tensor, img2_tensor), int(label)

    def __len__(self):
        """返回数据集大小"""
        return len(self.lines)

    @staticmethod
    def _load_image(file_path):
        """私有方法：加载图像并处理格式/错误"""
        try:
            with Image.open(file_path) as img:
                return img.convert('RGB')  # 强制转换为RGB三通道（处理PNG透明通道）
        except Exception as e:
            raise ValueError(f"图像加载失败: {file_path}, 错误: {str(e)}")
