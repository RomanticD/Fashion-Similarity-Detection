from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import pymysql
import os
from dotenv import load_dotenv

app = Flask(__name__)
# 允许所有路由跨域，如果你只想配置特定路由，也可以局部使用 @cross_origin()
CORS(app, resources={r"/*": {"origins": "*"}})

load_dotenv()

# --------------- 数据库连接 -------------------


def get_connection():
    """获取数据库连接"""
    connection = pymysql.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT")),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )
    return connection
