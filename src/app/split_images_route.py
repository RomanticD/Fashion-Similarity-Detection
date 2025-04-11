import base64
import logging

import numpy as np
from flask import request, app, jsonify, Flask, Blueprint, current_app
from flask_cors import cross_origin, CORS

from src.db.db_connect import get_connection
from src.repo.split_images_repo import create_split_image_db, read_split_image_db, update_split_image_db, \
    delete_split_image_db, save_to_db

from PIL import Image
import io

from src.utils.data_conversion import numpy_to_base64

# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api_sp = Blueprint('split_images', __name__)

CORS(api_sp)

@api_sp.route("/splitted_images", methods=["POST", "OPTIONS"])
@cross_origin()
def create_splitted_image():
    """
    创建子图记录
    前端传递 JSON:
    {
      "splitted_image_id": "xxx_sub1",
      "splitted_image_path": "/path/to/sub.jpg",
      "original_image_id": "xxx",  # 与 image_features(image_id) 对应
      "bounding_box": "100,50,200,300"
    }
    """
    data = request.get_json()
    splitted_image_id = data.get("splitted_image_id")
    splitted_image_path = data.get("splitted_image_path")
    original_image_id = data.get("original_image_id")
    bounding_box = data.get("bounding_box")

    if not splitted_image_id or not splitted_image_path or not original_image_id:
        return jsonify({"success": False, "message": "请填写完整信息！"}), 400

    success = create_split_image_db(splitted_image_id, splitted_image_path, original_image_id, bounding_box)
    if success:
        return jsonify({"success": True, "message": f"子图记录创建成功（splitted_image_id={splitted_image_id}）"})
    else:
        return jsonify({"success": False, "message": "创建子图失败，请检查后端日志"}), 500

@api_sp.route("/splitted_images/<string:splitted_image_id>", methods=["GET", "OPTIONS"])
@cross_origin()
def read_splitted_image(splitted_image_id):
    """
    查询子图记录
    URL: /splitted_images/<splitted_image_id>
    """
    result = read_split_image_db(splitted_image_id)
    if result:
        return jsonify({"success": True, "data": result})
    else:
        return jsonify({"success": False, "message": "未查询到对应子图记录"}), 404

@api_sp.route("/splitted_images/<string:splitted_image_id>", methods=["PUT", "OPTIONS"])
@cross_origin()
def update_splitted_image(splitted_image_id):
    """
    更新子图记录
    前端传递 JSON:
    {
      "new_path": "/new/path/to/sub.jpg",
      "new_bounding_box": "200,100,300,400"
    }
    """
    data = request.get_json()
    new_path = data.get("new_path")
    new_bounding_box = data.get("new_bounding_box")

    success = update_split_image_db(splitted_image_id, new_path, new_bounding_box)
    if success:
        return jsonify({"success": True, "message": f"子图记录更新成功（splitted_image_id={splitted_image_id}）"})
    else:
        return jsonify({"success": False, "message": "子图记录更新失败，请检查后端日志"}), 500

@api_sp.route("/splitted_images/<string:splitted_image_id>", methods=["DELETE", "OPTIONS"])
@cross_origin()
def delete_splitted_image(splitted_image_id):
    """
    删除子图记录
    URL: /splitted_images/<splitted_image_id>
    """
    success = delete_split_image_db(splitted_image_id)
    if success:
        return jsonify({"success": True, "message": f"子图记录已删除（splitted_image_id={splitted_image_id}）"})
    else:
        return jsonify({"success": False, "message": "子图记录删除失败，请检查后端日志"}), 500


@api_sp.route("/splitted_images/by_original/<string:original_image_id>", methods=["GET", "OPTIONS"])
@cross_origin()
def read_splitted_images_by_original(original_image_id):
    """
    根据 original_image_id 查询所有子图
    GET /splitted_images/by_original/<original_image_id>
    """
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            sql = "SELECT * FROM splitted_images WHERE original_image_id = %s"
            cursor.execute(sql, (original_image_id,))
            results = cursor.fetchall()  # fetchall 返回多行
        if results:
            return jsonify({"success": True, "data": results})
        else:
            return jsonify({"success": False, "message": "未查询到子图记录"}), 404
    except Exception as e:
        print("Error reading splitted_images by original:", e)
        return jsonify({"success": False, "message": "查询异常，请查看后端日志"}), 500
    finally:
        conn.close()

