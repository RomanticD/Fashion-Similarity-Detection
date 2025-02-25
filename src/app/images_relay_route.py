import json
import numpy as np
from flask import request, jsonify, Blueprint
from src.core.image_similarity import extract_feature, cosine_similarity
from src.repo.split_images_repo import select_all_vectors, select_image_data_by_id
from src.utils.data_conversion import base64_to_numpy

# 定义一个 Blueprint 来组织路由
api_rp = Blueprint('images_relay', __name__)

@api_rp.route("/relay_image", methods=["POST"])
def image_relay():
    try:
        # 获取传递的 JSON 数据
        data = request.get_json()
        num = data.get('num')  # 获取返回的条数
        base64_image = data.get('image_base64')  # 获取 Base64 图像数据

        if not base64_image:
            return jsonify({"error": "No image data found"}), 400

        # 将 Base64 字符串转换为 NumPy 数组
        image_np = base64_to_numpy(base64_image)

        # 提取图像特征
        image_feature = extract_feature(image_np)

        # 获取数据库中所有图像的向量
        rows = select_all_vectors()
        data = [{'id': row[0], 'vector': np.array(json.loads(row[1]))} for row in rows]

        # 计算与目标图像的相似度
        similarities = [
            (item['id'], cosine_similarity(image_feature, item['vector'])) for item in data
        ]

        # 按相似度降序排列
        similarities.sort(key=lambda x: x[1], reverse=True)

        result = []
        for idx, sim in similarities[:num]:
            # 根据 id 查询对应的图像数据
            base64_string = select_image_data_by_id(idx)

            result.append({
                "id": idx,
                "similarity": sim,
                "processed_image_base64": base64_string if base64_string else None  # 设置图像数据或 None
            })

        return jsonify(result)  # 返回 JSON 格式的结果

    except Exception as e:
        return jsonify({"error": str(e)}), 500