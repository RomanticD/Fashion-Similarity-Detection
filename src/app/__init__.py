from flask import Flask, request
from werkzeug.middleware.proxy_fix import ProxyFix

from .details_route import api_id
from .history_route import api_sh
from .images_relay_route import api_rp
from .images_route import api_bp
from .split_images_route import api_sp
from .image_upload_route import api_up
from .supabse_route import api_auth
from .image_processing_route import api_proc


def create_app():
    # 创建 Flask 应用实例
    app = Flask(__name__)

    # 添加 ProxyFix 中间件处理代理
    app.wsgi_app = ProxyFix(app.wsgi_app)

    # 配置应用
    app.config['PROPAGATE_EXCEPTIONS'] = True

    # 注册 Blueprint（将 routes.py 中的路由注册到应用中）
    app.register_blueprint(api_bp)
    app.register_blueprint(api_sp)
    app.register_blueprint(api_rp)
    app.register_blueprint(api_up)
    app.register_blueprint(api_auth)
    app.register_blueprint(api_id)
    app.register_blueprint(api_sh)
    app.register_blueprint(api_proc)

    # 创建一个请求前的钩子，设置超时响应
    @app.before_request
    def handle_chunking():
        request_is_streaming = 'Transfer-Encoding' in request.headers and \
                               request.headers['Transfer-Encoding'].lower() == 'chunked'
        if request_is_streaming:
            app.config['RESPONSE_TIMEOUT'] = 30

    return app