from flask import Flask
from flask_cors import CORS

# 创建 Flask 应用实例
app = Flask(__name__)

# 允许所有路由跨域
CORS(app, resources={r"/*": {"origins": "*"}})

# 导入路由模块
import split_images_route
import images_route