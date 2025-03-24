import base64
import json
import logging
import time
import numpy as np
from flask import request, jsonify, Blueprint
from flask_cors import cross_origin

from src.core.image_similarity import extract_feature, cosine_similarity
from src.db.db_connect import get_connection
from src.repo.split_images_repo import select_all_vectors, select_image_data_by_id
from src.utils.data_conversion import base64_to_numpy

# 定义一个 Blueprint 来组织路由
api_rp = Blueprint('images_relay', __name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def select_multiple_image_data_by_ids(ids):
    """
    Fetch multiple image records by their IDs

    Parameters:
    ids (list): List of image IDs to fetch

    Returns:
    dict: Dictionary mapping ID to image data
    """
    if not ids:
        return {}

    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            placeholders = ', '.join(['%s'] * len(ids))
            sql = f"SELECT id, splitted_image_data FROM splitted_images WHERE id IN ({placeholders})"
            cursor.execute(sql, ids)
            rows = cursor.fetchall()

            # Convert to dictionary for O(1) lookup
            result = {row['id']: row for row in rows}
            return result
    except Exception as e:
        print(f"Error fetching multiple image data: {e}")
        return {}
    finally:
        conn.close()

def batch_cosine_similarity(query_vector, all_vectors):
    """
    Calculate cosine similarity between one query vector and multiple vectors

    Parameters:
    query_vector (np.ndarray): Shape (n_features,)
    all_vectors (np.ndarray): Shape (n_samples, n_features)

    Returns:
    np.ndarray: Shape (n_samples,) containing similarity scores
    """
    # Normalize query vector
    query_norm = np.linalg.norm(query_vector)
    if query_norm == 0:
        return np.zeros(all_vectors.shape[0])
    query_normalized = query_vector / query_norm

    # Normalize all vectors (along axis 1)
    vectors_norm = np.linalg.norm(all_vectors, axis=1, keepdims=True)
    # Handle zero norm vectors to avoid division by zero
    vectors_norm[vectors_norm == 0] = 1.0
    vectors_normalized = all_vectors / vectors_norm

    # Calculate dot product which gives cosine similarity for normalized vectors
    similarities = np.dot(vectors_normalized, query_normalized)

    return similarities


@api_rp.route("/relay_image", methods=["POST"])
@cross_origin()
def image_relay():
    start_time = time.time()

    try:
        data = request.get_json()
        num = data.get('num', 5)  # Default to 5 if not specified
        base64_image = data.get('image_base64')

        if not base64_image:
            print("错误: 请求数据中没有 'image_base64' 字段")
            return jsonify({"error": "'image_base64' 字段是必需的"}), 400

        print(f"返回条数: {num}, 收到的 image_base64: {base64_image[:30]}...")

        # Image conversion
        convert_start = time.time()
        image_np = base64_to_numpy(base64_image)
        print(f"图像转换耗时: {time.time() - convert_start:.4f}秒")

        # Feature extraction
        feature_start = time.time()
        image_feature = extract_feature(image_np)

        # Database fetch
        db_start = time.time()
        rows = select_all_vectors()
        print(f"从数据库获取向量耗时: {time.time() - db_start:.4f}秒")
        print(f"共从数据库获取到 {len(rows)} 条向量数据")

        if not rows:
            return jsonify([])

        # Process vectors
        process_start = time.time()
        vector_arrays = []
        vector_ids = []

        for row in rows:
            try:
                vector_list = json.loads(row['vector'])
                vector_arrays.append(vector_list)
                vector_ids.append(row['id'])
            except Exception as e:
                print(f"处理记录 {row['id']} 时出错: {e}")

        vectors_matrix = np.array(vector_arrays)
        print(f"向量处理耗时: {time.time() - process_start:.4f}秒")

        # Calculate similarities
        sim_start = time.time()
        similarities = batch_cosine_similarity(image_feature, vectors_matrix)
        similarity_pairs = list(zip(vector_ids, similarities))
        similarity_pairs.sort(key=lambda x: x[1], reverse=True)
        top_similarities = similarity_pairs[:num]
        print(f"相似度计算和排序耗时: {time.time() - sim_start:.4f}秒")

        # Fetch image data in batch
        image_fetch_start = time.time()
        top_ids = [id for id, _ in top_similarities]

        # Add this function to split_images_repo.py
        all_image_data = select_multiple_image_data_by_ids(top_ids)

        result = []
        for idx, sim in top_similarities:
            image_data = all_image_data.get(idx)
            if image_data and 'splitted_image_data' in image_data:
                binary_string = image_data['splitted_image_data']
                base64_string = base64.b64encode(binary_string).decode("utf-8")

                result.append({
                    "id": idx,
                    "similarity": float(sim),  # Convert numpy float to Python float
                    "processed_image_base64": base64_string
                })
            else:
                print(f"未找到 ID 为 {idx} 的图像数据")
                result.append({
                    "id": idx,
                    "similarity": float(sim),
                    "processed_image_base64": None
                })

        print(f"获取图像数据耗时: {time.time() - image_fetch_start:.4f}秒")
        total_time = time.time() - start_time
        print(f"总运行时间: {total_time:.4f}秒")

        return jsonify(result)

    except Exception as e:
        total_time = time.time() - start_time
        print(f"发生错误! 总运行时间: {total_time:.4f}秒")
        print(f"错误详情: {e}")
        return jsonify({"error": str(e), "execution_time": total_time}), 500