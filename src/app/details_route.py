from flask import Blueprint
from flask import request, app, jsonify, Blueprint
from flask_cors import cross_origin, CORS

from src.repo.images_details_repo import read_image_detail

api_id = Blueprint('image_detail', __name__)
CORS(api_id)

@api_id.route("/image_detail", methods=["POST"])
@cross_origin()
def read_details():
    """
    查询子图记录
    URL: /images_detail/
    前端传入数据
    {
        'splitted_image_id':01_segment_10
        /未处理
    }
    """
    data = request.get_json()
    splitted_image_id = data.get('splitted_image_id')
    original_image_id = splitted_image_id.split("_")[0]
    result = read_image_detail(original_image_id)
    if result:
        return jsonify({"success": True, "message": result})
    else:
        return jsonify({"success": False, "message": "创建记录失败，检查后端日志"}), 500