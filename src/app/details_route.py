from flask import Blueprint
from flask import request, app, jsonify, Blueprint
from flask_cors import cross_origin, CORS

from src.repo.images_details_repo import read_image_detail

api_id = Blueprint('image_detail', __name__)

@api_id.route("/image_detail/<int:original_image_id>", methods=["GET"])
@cross_origin()
def read_details(original_image_id):
    """
    查询子图记录
    URL: /images_detail/<original_image_id>
    """
    result = read_image_detail(original_image_id)
    if result:
        return jsonify({"success": True, "message": result})
    else:
        return jsonify({"success": False, "message": "创建记录失败，检查后端日志"}), 500
    """
    查询结果: {'image_id': '01', 
    'store_name': '1', 
    'product_name': '1', 
    'product_link': '1', 
    'created_time': datetime.datetime(2025, 3, 1, 21, 15, 1), 
    'updated_time': datetime.datetime(1970, 1, 2, 0, 0)}
    """
