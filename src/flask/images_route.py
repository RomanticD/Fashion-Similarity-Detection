from flask import request, app, jsonify, Flask
from flask_cors import cross_origin, CORS

from src.repo.images_repo import create_image_feature_db, read_image_feature_db, update_image_feature_db, \
    delete_image_feature_db

@app.route("/image_features", methods=["POST", "OPTIONS"])
@cross_origin()
def create_record():
    """
    创建 image_features 记录
    前端传递 JSON:
    {
      "image_id": "...",
      "image_path": "...",
      "features": "..."
    }
    """
    data = request.get_json()
    image_id = data.get("image_id")
    image_path = data.get("image_path")
    features = data.get("features")

    if not image_id or not image_path or not features:
        return jsonify({"success": False, "message": "请填写完整信息！"}), 400

    success = create_image_feature_db(image_id, image_path, features)
    if success:
        return jsonify({"success": True, "message": f"创建记录成功（image_id={image_id}）"})
    else:
        return jsonify({"success": False, "message": "创建记录失败，检查后端日志"}), 500

@app.route("/image_features/<string:image_id>", methods=["GET", "OPTIONS"])
@cross_origin()
def read_record(image_id):
    """
    查询 image_features 记录
    URL: /image_features/<image_id>
    """
    result = read_image_feature_db(image_id)
    if result:
        return jsonify({"success": True, "data": result})
    else:
        return jsonify({"success": False, "message": "没有查询到相关记录"}), 404

@app.route("/image_features/<string:image_id>", methods=["PUT", "OPTIONS"])
@cross_origin()
def update_record(image_id):
    """
    更新 image_features 记录
    前端传递 JSON:
    {
      "new_path": "...",
      "new_features": "..."
    }
    """
    data = request.get_json()
    new_path = data.get("new_path")
    new_features = data.get("new_features")

    if not new_path or not new_features:
        return jsonify({"success": False, "message": "请填写完整信息！"}), 400

    success = update_image_feature_db(image_id, new_path, new_features)
    if success:
        return jsonify({"success": True, "message": f"更新记录成功（image_id={image_id}）"})
    else:
        return jsonify({"success": False, "message": "更新记录失败，检查后端日志"}), 500

@app.route("/image_features/<string:image_id>", methods=["DELETE", "OPTIONS"])
@cross_origin()
def delete_record(image_id):
    """
    删除 image_features 记录
    URL: /image_features/<image_id>
    """
    success = delete_image_feature_db(image_id)
    if success:
        return jsonify({"success": True, "message": f"删除记录成功（image_id={image_id}）"})
    else:
        return jsonify({"success": False, "message": "删除记录失败，检查后端日志"}), 500
