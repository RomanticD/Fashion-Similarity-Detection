import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# ---------------------
# 1) 加载预训练的 ResNet50 模型，并去掉最终分类层
# ---------------------
model = models.resnet50(pretrained=True)
model.eval()  # 设置为推理模式
model.fc = torch.nn.Identity()  # 去掉最后的分类层，保留 2048 维特征

# ---------------------
# 2) 定义预处理Transforms
# ---------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def extract_feature(img_path):
    """
    给定图片路径，返回提取的 ResNet50 特征（2048 维向量）
    """
    img = Image.open(img_path).convert('RGB')
    img_t = transform(img)
    img_t = img_t.unsqueeze(0)  # 增加batch维度

    with torch.no_grad():
        feat = model(img_t)  # 输出形状为 [1, 2048]
    return feat.squeeze(0).numpy()  # 转换为形状 [2048] 的numpy数组


def cosine_similarity(vec1, vec2):
    """
    计算两个向量的余弦相似度
    """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


def load_images_feature_vectors(img_paths):
    """
    传入一组图片路径，返回一个字典，key 为图片路径，value 为对应的特征向量
    """
    features_dict = {}
    for path in img_paths:
        features_dict[path] = extract_feature(path)
    return features_dict


def load_single_image_feature_vector(img_path):
    """
    传入单张图片路径，返回一个仅包含该图片名称和特征向量的字典
    """
    return {img_path: extract_feature(img_path)}


def compare_similarities(single_image_dict, images_dict):
    """
    对比 single_image_dict 中的图片与 images_dict 中所有图片的相似度
    返回一个列表，列表中的每个元素包含 (图片路径, 相似度)
    """
    # single_image_dict 里只会有一个 key
    single_img_path, single_img_vector = list(single_image_dict.items())[0]

    results = []
    for img_path, img_vector in images_dict.items():
        sim = cosine_similarity(single_img_vector, img_vector)
        results.append((img_path, sim))
    return results


if __name__ == '__main__':
    # 示例：现有图片
    existing_imgs = ["Assets/testCoat2.png", "Assets/testCoat3.png"]
    # 待对比的单张图片
    single_img = "Assets/testCoat.png"

    # 1) 加载已有图片的特征
    images_dict = load_images_feature_vectors(existing_imgs)

    # 2) 加载单张需要比较的图片特征
    single_image_dict = load_single_image_feature_vector(single_img)

    # 3) 计算相似度
    similarity_results = compare_similarities(single_image_dict, images_dict)

    # 打印结果
    print(f"待对比图片: {single_img}")
    for img_path, sim in similarity_results:
        print(f"图片 {img_path} 的相似度: {sim}")