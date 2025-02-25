from flask import current_app

from src.db.db_connect import get_connection
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_split_image_db(splitted_image_id, splitted_image_path, original_image_id, bounding_box=None):
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

def read_split_image_db(splitted_image_id):
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

def update_split_image_db(splitted_image_id, new_path=None, new_bounding_box=None):
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

def delete_split_image_db(splitted_image_id):
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

def save_to_db(splitted_image_id, splitted_image_path, original_image_id, bounding_box, binary_data, vector):
    # 连接到数据库
    conn = get_connection()  # 你可能需要在这里配置数据库连接
    cursor = conn.cursor()

    try:
        # 插入图像数据和特征到数据库
        cursor.execute("""
            INSERT INTO splitted_images (splitted_image_id, splitted_image_path, original_image_id, bounding_box, splitted_image_data, vector)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (splitted_image_id, splitted_image_path, original_image_id, bounding_box, binary_data, vector))

        # 提交事务
        conn.commit()
        logger.info(f"Image {splitted_image_id} uploaded to the database along with its feature.")
    except Exception as e:
        logger.error(f"Error uploading image to the database: {e}")
        conn.rollback()
    finally:
        # 关闭数据库连接
        cursor.close()
        conn.close()

def select_all_vectors():
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT id, vector FROM splitted_images")
            rows = cursor.fetchall()
        conn.commit()
        return rows
    except Exception as e:
        print("Error select:", e)
        conn.rollback()
        return False
    finally:
        conn.close()

def select_image_data_by_id(id):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT splitted_image_data FROM splitted_images WHERE id = %s", (id,))
            row = cursor.fetchone()  # 获取单条记录
            if row:
                return row[0]  # 返回图像数据（第一列数据）
            else:
                return None  # 如果没有找到该 id 对应的记录，返回 None
    except Exception as e:
        current_app.logger.error(f"Error selecting image data by id {id}: {str(e)}")
        conn.rollback()
        return None
    finally:
        conn.close()
