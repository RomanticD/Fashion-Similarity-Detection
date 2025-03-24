import base64
import json
import logging
import time
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
    # 记录函数开始时间
    start_time = time.time()

    try:
        # 获取传递的 JSON 数据
        data = request.get_json()

        num = data.get('num')  # 获取返回的条数
        base64_image = data.get('image_base64')  # 获取 Base64 图像数据
        if not base64_image:
            print("错误: 请求数据中没有 'image_base64' 字段")
            return jsonify({"error": "'image_base64' 字段是必需的"}), 400

        print(f"返回条数: {num}, 收到的 image_base64: {base64_image[:30]}...")

        # 记录图像转换开始时间
        convert_start = time.time()

        # 将 Base64 字符串转换为 NumPy 数组
        image_np = base64_to_numpy(base64_image)
        print(f"图像转换耗时: {time.time() - convert_start:.4f}秒")

        # 记录特征提取开始时间
        feature_start = time.time()

        # 提取图像特征
        image_feature = extract_feature(image_np)
        print(f"特征提取耗时: {time.time() - feature_start:.4f}秒")

        # 记录向量获取开始时间
        db_start = time.time()

        # 获取数据库中所有图像的向量
        rows = select_all_vectors()
        print(f"从数据库获取向量耗时: {time.time() - db_start:.4f}秒")
        print(f"共从数据库获取到 {len(rows)} 条向量数据")

        # 记录向量处理开始时间
        process_start = time.time()

        # 批量处理向量数据
        vectors_data = []
        for row in rows:
            try:
                vector_list = json.loads(row['vector'])
                vectors_data.append({
                    'id': row['id'],
                    'vector': np.array(vector_list)
                })
            except Exception as e:
                print(f"处理记录 {row['id']} 时出错: {e}")

        print(f"向量处理耗时: {time.time() - process_start:.4f}秒")

        # 记录相似度计算开始时间
        sim_start = time.time()

        # 使用向量化操作计算相似度
        similarities = []
        for item in vectors_data:
            sim = cosine_similarity(image_feature, item['vector'])
            similarities.append((item['id'], sim))

        # 按相似度降序排列
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 只保留前 num 个结果
        top_similarities = similarities[:num]
        print(f"相似度计算和排序耗时: {time.time() - sim_start:.4f}秒")

        # 记录图像获取开始时间
        image_fetch_start = time.time()

        # 获取相似图像的 ID 列表
        top_ids = [id for id, _ in top_similarities]

        # 创建结果列表
        result = []
        for idx, sim in top_similarities:
            # 根据 id 查询对应的图像数据
            image_data = select_image_data_by_id(idx)
            if image_data and 'splitted_image_data' in image_data:
                binary_string = image_data['splitted_image_data']
                base64_string = base64.b64encode(binary_string).decode("utf-8")

                result.append({
                    "id": idx,
                    "similarity": sim,
                    "processed_image_base64": base64_string
                })
            else:
                print(f"未找到 ID 为 {idx} 的图像数据")
                result.append({
                    "id": idx,
                    "similarity": sim,
                    "processed_image_base64": None
                })

        print(f"获取图像数据耗时: {time.time() - image_fetch_start:.4f}秒")

        # 计算总运行时间
        total_time = time.time() - start_time
        print(f"总运行时间: {total_time:.4f}秒")

        return jsonify(result)

    except Exception as e:
        # 计算总运行时间（即使发生错误）
        total_time = time.time() - start_time
        print(f"发生错误! 总运行时间: {total_time:.4f}秒")
        print(f"错误详情: {e}")
        return jsonify({"error": str(e), "execution_time": total_time}), 500