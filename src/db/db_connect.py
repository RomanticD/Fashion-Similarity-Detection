from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import pymysql
import json
import base64
import uuid
import os

app = Flask(__name__)
# 允许所有路由跨域，如果你只想配置特定路由，也可以局部使用 @cross_origin()
CORS(app, resources={r"/*": {"origins": "*"}})

# --------------- 数据库连接 -------------------
def get_connection():
    """获取数据库连接"""
    connection = pymysql.connect(
        host="rm-cn-gh645bd4t000n17o.rwlb.rds.aliyuncs.com",
        port=3306,
        user="fashion",
        password="Fashion123",
        database="fashion_db",
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )
    return connection

# --------------- image_features CRUD -------------------
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
            sql = "SELECT * FROM image_features WHERE image_id = %s"
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
            sql = "DELETE FROM image_features WHERE image_id = %s"
            cursor.execute(sql, (image_id,))
        conn.commit()
        return True
    except Exception as e:
        print("Error deleting record:", e)
        conn.rollback()
        return False
    finally:
        conn.close()

# --------------- splitted_images CRUD -------------------
def create_splitted_image_db(splitted_image_id, splitted_image_path, original_image_id, bounding_box=None):
    """
    向 splitted_images 表插入一条记录。
    """
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            sql = """
                INSERT INTO splitted_images
                    (splitted_image_id, splitted_image_path, original_image_id, bounding_box)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(sql, (splitted_image_id, splitted_image_path, original_image_id, bounding_box))
        conn.commit()
        return True
    except Exception as e:
        print("Error creating splitted_image:", e)
        conn.rollback()
        return False
    finally:
        conn.close()

def read_splitted_image_db(splitted_image_id):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            sql = "SELECT * FROM splitted_images WHERE splitted_image_id = %s"
            cursor.execute(sql, (splitted_image_id,))
            result = cursor.fetchone()
        return result
    except Exception as e:
        print("Error reading splitted_image:", e)
        return None
    finally:
        conn.close()

def update_splitted_image_db(splitted_image_id, new_path=None, new_bounding_box=None):
    conn = get_connection()
    try:
        fields = []
        values = []
        if new_path is not None:
            fields.append("splitted_image_path = %s")
            values.append(new_path)
        if new_bounding_box is not None:
            fields.append("bounding_box = %s")
            values.append(new_bounding_box)

        if not fields:  # 没有要更新的字段
            return False

        set_clause = ", ".join(fields)
        sql = f"UPDATE splitted_images SET {set_clause} WHERE splitted_image_id = %s"
        values.append(splitted_image_id)

        with conn.cursor() as cursor:
            cursor.execute(sql, values)
        conn.commit()
        return True
    except Exception as e:
        print("Error updating splitted_image:", e)
        conn.rollback()
        return False
    finally:
        conn.close()

def delete_splitted_image_db(splitted_image_id):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            sql = "DELETE FROM splitted_images WHERE splitted_image_id = %s"
            cursor.execute(sql, (splitted_image_id,))
        conn.commit()
        return True
    except Exception as e:
        print("Error deleting splitted_image:", e)
        conn.rollback()
        return False
    finally:
        conn.close()

# --------------- flask 路由 ---------------
# -- 1) image_features 路由

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

    if not new_path and not new_features:
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

# -- 2) splitted_images 路由

@app.route("/splitted_images", methods=["POST", "OPTIONS"])
@cross_origin()
def create_splitted_image():
    """
    创建子图记录
    前端传递 JSON:
    {
      "splitted_image_id": "xxx_sub1",
      "splitted_image_path": "/path/to/sub.jpg",
      "original_image_id": "xxx",  # 与 image_features(image_id) 对应
      "bounding_box": "100,50,200,300"
    }
    """
    data = request.get_json()
    splitted_image_id = data.get("splitted_image_id")
    splitted_image_path = data.get("splitted_image_path")
    original_image_id = data.get("original_image_id")
    bounding_box = data.get("bounding_box")

    if not splitted_image_id or not splitted_image_path or not original_image_id:
        return jsonify({"success": False, "message": "请填写完整信息！"}), 400

    success = create_splitted_image_db(splitted_image_id, splitted_image_path, original_image_id, bounding_box)
    if success:
        return jsonify({"success": True, "message": f"子图记录创建成功（splitted_image_id={splitted_image_id}）"})
    else:
        return jsonify({"success": False, "message": "创建子图失败，请检查后端日志"}), 500

@app.route("/splitted_images/<string:splitted_image_id>", methods=["GET", "OPTIONS"])
@cross_origin()
def read_splitted_image(splitted_image_id):
    """
    查询子图记录
    URL: /splitted_images/<splitted_image_id>
    """
    result = read_splitted_image_db(splitted_image_id)
    if result:
        return jsonify({"success": True, "data": result})
    else:
        return jsonify({"success": False, "message": "未查询到对应子图记录"}), 404

@app.route("/splitted_images/<string:splitted_image_id>", methods=["PUT", "OPTIONS"])
@cross_origin()
def update_splitted_image(splitted_image_id):
    """
    更新子图记录
    前端传递 JSON:
    {
      "new_path": "/new/path/to/sub.jpg",
      "new_bounding_box": "200,100,300,400"
    }
    """
    data = request.get_json()
    new_path = data.get("new_path")
    new_bounding_box = data.get("new_bounding_box")

    success = update_splitted_image_db(splitted_image_id, new_path, new_bounding_box)
    if success:
        return jsonify({"success": True, "message": f"子图记录更新成功（splitted_image_id={splitted_image_id}）"})
    else:
        return jsonify({"success": False, "message": "子图记录更新失败，请检查后端日志"}), 500

@app.route("/splitted_images/<string:splitted_image_id>", methods=["DELETE", "OPTIONS"])
@cross_origin()
def delete_splitted_image(splitted_image_id):
    """
    删除子图记录
    URL: /splitted_images/<splitted_image_id>
    """
    success = delete_splitted_image_db(splitted_image_id)
    if success:
        return jsonify({"success": True, "message": f"子图记录已删除（splitted_image_id={splitted_image_id}）"})
    else:
        return jsonify({"success": False, "message": "子图记录删除失败，请检查后端日志"}), 500

@app.route("/splitted_images/by_original/<string:original_image_id>", methods=["GET", "OPTIONS"])
def read_splitted_images_by_original(original_image_id):
    """
    根据 original_image_id 查询所有子图
    GET /splitted_images/by_original/<original_image_id>
    """
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            sql = "SELECT * FROM splitted_images WHERE original_image_id = %s"
            cursor.execute(sql, (original_image_id,))
            results = cursor.fetchall()  # fetchall 返回多行
        if results:
            return jsonify({"success": True, "data": results})
        else:
            return jsonify({"success": False, "message": "未查询到子图记录"}), 404
    except Exception as e:
        print("Error reading splitted_images by original:", e)
        return jsonify({"success": False, "message": "查询异常，请查看后端日志"}), 500
    finally:
        conn.close()

# --------------- 上传 Base64 图片 -------------------

# 指定图片存储目录
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # 确保文件夹存在
@app.route("/upload", methods=["POST"])
def upload_image():
    print("收到上传请求")  # 先检查 Flask 是否收到请求
    print("📂 图片存储目录:", UPLOAD_FOLDER)
    try:
        data = request.json
        base64_image = data.get("image")

        if not base64_image:
            return jsonify({"message": "没有收到图片数据", "success": False}), 400

        # 解析 Base64 数据
        try:
            header, encoded = base64_image.split(",", 1)  # 去掉 "data:image/png;base64,"
            file_extension = header.split("/")[1].split(";")[0]  # 提取文件扩展名 (png, jpg)
            image_data = base64.b64decode(encoded)
        except Exception as e:
            return jsonify({"message": f"Base64 解析错误: {str(e)}", "success": False}), 400

        # 生成唯一 image_id 和文件名
        image_id = str(uuid.uuid4())[:8]  # 生成8位唯一 ID
        image_filename = f"{image_id}.{file_extension}"
        image_path = os.path.join(UPLOAD_FOLDER, image_filename)

        # 保存图片到 uploads 目录
        try:
            with open(image_path, "wb") as f:
                f.write(image_data)
            print(f"图片保存成功: {image_path}")  # 确保代码执行
        except IOError as e:
            print(f"图片保存失败: {str(e)}")
            return jsonify({"message": f"文件存储失败: {str(e)}", "success": False}), 500


# 存入数据库
        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                sql = """
                    INSERT INTO image_features (image_id, image_path, features)
                    VALUES (%s, %s, %s)
                """
                cursor.execute(sql, (image_id, image_path, ""))  # 用 "" 代替 NULL
                # cursor.execute(sql, (image_id, image_path, None))  # features 暂时设为空
            conn.commit()
        finally:
            conn.close()

        return jsonify({
            "message": "图片上传成功",
            "image_id": image_id,
            "path": image_path,
            "success": True
        })

    except Exception as e:
        return jsonify({"message": f"上传失败: {str(e)}", "success": False}), 500


# --------------- 主程序入口 ---------------
if __name__ == "__main__":
    # 可根据需要修改 host / port
    app.run(host="0.0.0.0", port=5001, debug=True)