"""import json
import numpy as np
from flask import request, jsonify, Blueprint
from flask_cors import cross_origin

from src.core.image_similarity import extract_feature, cosine_similarity
from src.repo.split_images_repo import select_all_vectors, select_image_data_by_id
from src.utils.data_conversion import base64_to_numpy

# 定义一个 Blueprint 来组织路由
api_rp = Blueprint('images_relay', __name__)

@api_rp.route("/relay_image", methods=["POST"])
@cross_origin()
def image_relay():
    try:
        # 获取传递的 JSON 数据
        data = request.get_json()
        num = data.get('num')  # 获取返回的条数
        base64_image = data.get('image_base64')  # 获取 Base64 图像数据
        print(f"Received data: {data}")  # 打印接收到的数据

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
"""
import base64
import json
import logging

import numpy as np
from flask import request, jsonify, Blueprint
from flask_cors import cross_origin

from src.core.image_similarity import extract_feature, cosine_similarity
from src.repo.split_images_repo import select_all_vectors, select_image_data_by_id
from src.utils.data_conversion import base64_to_numpy

# 定义一个 Blueprint 来组织路由
api_rp = Blueprint('images_relay', __name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@api_rp.route("/relay_image", methods=["POST"])
@cross_origin()
def image_relay():
    try:
        # 获取传递的 JSON 数据
        data = request.get_json()
        print(f"Received JSON data: {data}")  # 打印接收到的数据

        num = data.get('num')  # 获取返回的条数
        base64_image = data.get('image_base64')  # 获取 Base64 图像数据
        if not base64_image:
            print("Error: No 'image_base64' field in the request data.")
            return jsonify({"error": "'image_base64' field is required"}), 400  # 返回统一的错误消息

        print(f"Extracted num: {num}, image_base64: {base64_image[:30]}...")  # 只打印部分 Base64 图像数据

        # 将 Base64 字符串转换为 NumPy 数组
        image_np = base64_to_numpy(base64_image)
        print("Image successfully converted to NumPy array.")

        # 提取图像特征
        image_feature = extract_feature(image_np)
        print(f"Extracted image feature: {image_feature[:10]}...")  # 只打印部分特征数据

        # 获取数据库中所有图像的向量
        rows = select_all_vectors()
        print(f"Fetched {len(rows)} vectors from the database.")

        # data = [{'id': row[0], 'vector': np.array(json.loads(row[1]))} for row in rows]
        data = []
        print(data)
        for row in rows:
            print("1")
            try:
                print("2")
                print(row)
                vector_str = row['vector']  # 提取字符串
                print(vector_str)
                vector_list = json.loads(vector_str)
                print("提取成功：", vector_list)
                vector_array = np.array(vector_list)
                print("1")
                data.append({'id': row['id'], 'vector': vector_array})
                print("2")
            except Exception as e:
                print(f"Error processing row {row[0]}: {e}")
        print(f"Processed database vectors into list of dictionaries.")

        # 计算与目标图像的相似度
        similarities = [
            (item['id'], cosine_similarity(image_feature, item['vector'])) for item in data
        ]
        print(f"Computed similarities for {len(similarities)} images.")

        # 按相似度降序排列
        similarities.sort(key=lambda x: x[1], reverse=True)
        print("Sorted similarities in descending order.")

        result = []
        for idx, sim in similarities[:num]:
            print(f"Processing image with id: {idx} and similarity: {sim:.4f}")
            # 根据 id 查询对应的图像数据
            binary_string = select_image_data_by_id(idx)['splitted_image_data']
            print("1")
            base64_string = base64.b64encode(binary_string).decode("utf-8")
            '''base64_string = select_image_data_by_id(idx)'''

            if base64_string:
                print(f"Found base64 image data for id {idx}.")
            else:
                print(f"No image data found for id {idx}.")

            result.append({
                "id": idx,
                "similarity": sim,
                "processed_image_base64": base64_string if base64_string else None  # 设置图像数据或 None
            })

        print(f"Returning {len(result)} results.")
        print(jsonify(result))
        logger.info(f"Response JSON: {jsonify(result).get_data(as_text=True)}")
        return jsonify(result)

    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({"error": str(e)}), 500