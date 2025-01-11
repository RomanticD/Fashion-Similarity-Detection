from flask import Flask, request, jsonify
from flask_cors import CORS
import pymysql

app = Flask(__name__)
CORS(app)  # 允许跨域请求，方便调用

# ------------------ 数据库连接 ------------------

def get_connection():
    """获取数据库连接"""
    connection = pymysql.connect(
        host="database-1.c5282akgwxld.ap-southeast-2.rds.amazonaws.com",
        port=3306,
        user="admin",
        password="112345678",
        database="fashion_db",
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )
    return connection

# ------------------ CRUD 操作函数 ------------------

def create_image_feature_db(image_id, image_path, features):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            sql = """
                INSERT INTO image_features (image_id, image_path, features)
                VALUES (%s, %s, %s)
            """
            cursor.execute(sql, (image_id, image_path, features))
        conn.commit()
        return True
    except Exception as e:
        print("Error creating record:", e)
        conn.rollback()
        return False
    finally:
        conn.close()

def read_image_feature_db(image_id):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            sql = """
                SELECT * FROM image_features WHERE image_id = %s
            """
            cursor.execute(sql, (image_id,))
            result = cursor.fetchone()
        return result
    except Exception as e:
        print("Error reading record:", e)
        return None
    finally:
        conn.close()

def update_image_feature_db(image_id, new_path, new_features):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            sql = """
                UPDATE image_features
                SET image_path = %s,
                    features = %s
                WHERE image_id = %s
            """
            cursor.execute(sql, (new_path, new_features, image_id))
        conn.commit()
        return True
    except Exception as e:
        print("Error updating record:", e)
        conn.rollback()
        return False
    finally:
        conn.close()

def delete_image_feature_db(image_id):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            sql = """
                DELETE FROM image_features WHERE image_id = %s
            """
            cursor.execute(sql, (image_id,))
        conn.commit()
        return True
    except Exception as e:
        print("Error deleting record:", e)
        conn.rollback()
        return False
    finally:
        conn.close()

# ------------------ Flask 路由 ------------------

@app.route("/")
def index():
    return "Flask Server is running..."

@app.route("/image_features", methods=["POST"])
def create_record():
    """
    创建记录
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

@app.route("/image_features/<string:image_id>", methods=["GET"])
def read_record(image_id):
    """
    查询记录
    URL: /image_features/<image_id>
    """
    result = read_image_feature_db(image_id)
    if result:
        return jsonify({"success": True, "data": result})
    else:
        return jsonify({"success": False, "message": "没有查询到相关记录"}), 404

@app.route("/image_features/<string:image_id>", methods=["PUT"])
def update_record(image_id):
    """
    更新记录
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

@app.route("/image_features/<string:image_id>", methods=["DELETE"])
def delete_record(image_id):
    """
    删除记录
    URL: /image_features/<image_id>
    """
    success = delete_image_feature_db(image_id)
    if success:
        return jsonify({"success": True, "message": f"删除记录成功（image_id={image_id}）"})
    else:
        return jsonify({"success": False, "message": "删除记录失败，检查后端日志"}), 500

# ------------------ 主程序入口 ------------------

if __name__ == "__main__":
    # 可修改 host 和 port 用于本地部署
    app.run(host="0.0.0.0", port=5001, debug=True)