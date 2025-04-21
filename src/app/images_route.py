# images_route.py
from flask import request, app, jsonify, Blueprint
from flask_cors import cross_origin, CORS
import logging
import time

from src.repo.images_repo import create_splitted_image_db, read_splitted_image_db, update_splitted_image_db, \
    delete_split_image_db, get_splitted_images_by_original_id
from src.core.vector_index import VectorIndex

# 定义一个 Blueprint 来组织路由，__init__文件注册Blueprint
api_bp = Blueprint('images_route', __name__)

# 允许跨域
CORS(api_bp)

# 添加日志记录
logger = logging.getLogger(__name__)

# 初始化向量索引
vector_index = VectorIndex()

@api_bp.route("/splitted_images", methods=["POST", "OPTIONS"])
@cross_origin()
def create_record():
    """
    创建 splitted_images 记录
    前端传递 JSON:
    {
      "splitted_image_id": "...",
      "splitted_image_path": "...",
      "original_image_id": "...",
      "bounding_box": "...",      // 可选
      "vector": {},               // 可选
      "splitted_image_data": "..."// 可选，base64编码的图像数据
    }
    """
    data = request.get_json()
    splitted_image_id = data.get("splitted_image_id")
    splitted_image_path = data.get("splitted_image_path")
    original_image_id = data.get("original_image_id")
    bounding_box = data.get("bounding_box")
    vector = data.get("vector")
    
    # 处理可能的base64图像数据
    splitted_image_data = data.get("splitted_image_data")
    
    if not splitted_image_id or not splitted_image_path or not original_image_id:
        return jsonify({"success": False, "message": "请填写必要信息（splitted_image_id, splitted_image_path, original_image_id）！"}), 400

    # 调用方法
    success = create_splitted_image_db(
        splitted_image_id, 
        splitted_image_path, 
        original_image_id, 
        bounding_box, 
        vector, 
        splitted_image_data
    )

    if success:
        return jsonify({"success": True, "message": f"创建记录成功（splitted_image_id={splitted_image_id}）"})
    else:
        return jsonify({"success": False, "message": "创建记录失败，检查后端日志"}), 500

@api_bp.route("/splitted_images/<string:splitted_image_id>", methods=["GET", "OPTIONS"])
@cross_origin()
def read_record(splitted_image_id):
    """
    查询 splitted_images 记录
    URL: /splitted_images/<splitted_image_id>
    """
    result = read_splitted_image_db(splitted_image_id)
    if result:
        return jsonify({"success": True, "data": result})
    else:
        return jsonify({"success": False, "message": "没有查询到相关记录"}), 404

@api_bp.route("/splitted_images/by_original/<string:original_image_id>", methods=["GET", "OPTIONS"])
@cross_origin()
def get_by_original_id(original_image_id):
    """
    根据原图ID查询所有分割图片记录
    URL: /splitted_images/by_original/<original_image_id>
    """
    result = get_splitted_images_by_original_id(original_image_id)
    if result:
        return jsonify({"success": True, "data": result})
    else:
        return jsonify({"success": False, "message": "没有查询到相关记录"}), 404

@api_bp.route("/splitted_images/<string:splitted_image_id>", methods=["PUT", "OPTIONS"])
@cross_origin()
def update_record(splitted_image_id):
    """
    更新 splitted_images 记录
    前端传递 JSON:
    {
      "splitted_image_path": "...",  // 可选
      "original_image_id": "...",    // 可选
      "bounding_box": "...",         // 可选
      "vector": {},                  // 可选
      "splitted_image_data": "..."   // 可选，base64编码的图像数据
    }
    """
    data = request.get_json()
    new_path = data.get("splitted_image_path")
    new_original_id = data.get("original_image_id")
    new_box = data.get("bounding_box")
    new_vector = data.get("vector")
    new_image_data = data.get("splitted_image_data")

    # 至少需要一个更新字段
    if not any([new_path, new_original_id, new_box, new_vector, new_image_data]):
        return jsonify({"success": False, "message": "请至少提供一个需要更新的字段！"}), 400

    success = update_splitted_image_db(
        splitted_image_id, 
        new_path, 
        new_original_id, 
        new_box, 
        new_vector, 
        new_image_data
    )
    
    if success:
        return jsonify({"success": True, "message": f"更新记录成功（splitted_image_id={splitted_image_id}）"})
    else:
        return jsonify({"success": False, "message": "更新记录失败，检查后端日志"}), 500

@api_bp.route("/splitted_images/<string:splitted_image_id>", methods=["DELETE", "OPTIONS"])
@cross_origin()
def delete_record(splitted_image_id):
    """
    删除 splitted_images 记录
    URL: /splitted_images/<splitted_image_id>
    """
    logger.info(f"正在删除图片记录: {splitted_image_id}")
    success = delete_split_image_db(splitted_image_id)
    
    if not success:
        logger.error(f"从数据库中删除记录失败: {splitted_image_id}")
        return jsonify({"success": False, "message": "删除记录失败，检查后端日志"}), 500
        
    # 删除成功后，确保索引更新
    logger.info(f"成功从数据库删除记录，正在更新索引: {splitted_image_id}")
    
    rebuild_start_time = time.time()
    try:
        # 确保索引正确更新
        # 1. 先删除旧索引文件
        if vector_index.index_file.exists():
            vector_index.index_file.unlink()
            logger.info(f"已删除旧索引文件: {vector_index.index_file}")
            
        if vector_index.id_map_file.exists():
            vector_index.id_map_file.unlink()
            logger.info(f"已删除旧ID映射文件: {vector_index.id_map_file}")
        
        # 2. 强制重建索引
        index_result = vector_index.rebuild_index()
        rebuild_time = time.time() - rebuild_start_time
        
        if index_result[0] is None:
            logger.warning(f"索引更新失败，用时: {rebuild_time:.2f}秒")
            return jsonify({
                "success": True, 
                "message": f"删除记录成功（splitted_image_id={splitted_image_id}），但索引更新失败"
            })
        
        # 3. 清除内存中的旧索引实例，确保下次搜索读取新索引
        vector_index.index = None
        vector_index.ids = None
        vector_index.vectors = None
        
        logger.info(f"索引重建完成，用时: {rebuild_time:.2f}秒")
    except Exception as e:
        logger.error(f"索引更新错误: {e}")
        return jsonify({
            "success": True, 
            "message": f"删除记录成功（splitted_image_id={splitted_image_id}），但索引更新失败: {str(e)}"
        })
    
    logger.info(f"图片记录删除全部完成: {splitted_image_id}")
    return jsonify({
        "success": True, 
        "message": f"删除记录成功（splitted_image_id={splitted_image_id}），索引已更新"
    })


