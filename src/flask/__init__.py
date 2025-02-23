from flask import Flask
from flask_cors import CORS
import logging

# 导入路由模块
from .split_images_route import create_splitted_image, read_splitted_image, update_splitted_image, \
    delete_splitted_image, read_splitted_images_by_original, upload_splitted_image_to_db

# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    """
    创建并初始化 Flask 应用。
    """
    app = Flask(__name__)
    CORS(app)  # 启用 CORS

    # 注册路由
    app.add_url_rule('/splitted_images', view_func=create_splitted_image, methods=["POST", "OPTIONS"])
    app.add_url_rule('/splitted_images/<string:splitted_image_id>', view_func=read_splitted_image, methods=["GET", "OPTIONS"])
    app.add_url_rule('/splitted_images/<string:splitted_image_id>', view_func=update_splitted_image, methods=["PUT", "OPTIONS"])
    app.add_url_rule('/splitted_images/<string:splitted_image_id>', view_func=delete_splitted_image, methods=["DELETE", "OPTIONS"])
    app.add_url_rule('/splitted_images/by_original/<string:original_image_id>', view_func=read_splitted_images_by_original, methods=["GET", "OPTIONS"])
    app.add_url_rule('/split_image_upload', view_func=upload_splitted_image_to_db, methods=["POST"])

    return app