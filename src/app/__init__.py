from flask import Flask
from flask_cors import CORS

def create_app():
    # 创建 Flask 应用实例
    app = Flask(__name__)

    # 允许所有路由跨域
    CORS(app, resources={r"/*": {"origins": "*"}})

    return app