import base64
import os
from io import BytesIO

from PIL import Image
from flask import request, app, jsonify, Blueprint
from flask_cors import cross_origin, CORS

from src.core.image_similarity import extract_feature
from src.repo.images_repo import create_image_feature_db, read_image_feature_db, update_image_feature_db, \
    delete_image_feature_db
from src.utils.base64_to_numpy import base64_to_numpy

# 定义一个 Blueprint 来组织路由
api_rp = Blueprint('images_relay', __name__)

@api_rp.route("/relay_image", methods=["POST"])
def image_relay():
    try:
        # 获取传递的 JSON 数据
        data = request.get_json()

        num_return = data.get('num')

        # 从 JSON 数据中提取 Base64 字符串
        base64_image = data.get('image')  # Base64 字符串在 'image' 字段

        if not base64_image:
            return jsonify({"error": "No image data found"}), 400

        image_np = base64_to_numpy(base64_image)

        image_feature = extract_feature(image_np)





        """
        base64转numpy
        图像调用获取特征值
        特征值对比
        返回数据库最相似的
        



        """

        return jsonify({"success": True, "message": f"message -> numpy array"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500