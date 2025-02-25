from flask import Flask

from .images_relay_route import api_rp
from .images_route import api_bp
from .split_images_route import api_sp

def create_app():
    # 创建 Flask 应用实例
    app = Flask(__name__)

    # 注册 Blueprint（将 routes.py 中的路由注册到应用中）
    app.register_blueprint(api_bp)
    app.register_blueprint(api_sp)
    app.register_blueprint(api_rp)

    return app
