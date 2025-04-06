from flask import Blueprint
from flask import request, app, jsonify, Blueprint
from flask_cors import cross_origin, CORS

from src.repo.images_details_repo import read_image_detail

api_id = Blueprint('image_detail', __name__)

@api_id.route("/image_detail", methods=["GET"])
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
    ''' 
    查询结果:sample_data = {
    "image_id": 123,
    "store_name": "怪兽工作室",
    "brand": "unknown",
    "product_name": "新款长袖......",
    "store_description": "...",
    "url": "Invalid",
    "rating": 4.7,
    "tags": ["米线", "小吃", "环境好"],
    "sale_status": "on_sale",  # 假设是 on_sale / sold_out / coming_soon 等
    "size": ["S", "M", "L"],
    "waist_type": "high",
    "listing_season": "2024Spring",
    "season": "summer"
}

    '''