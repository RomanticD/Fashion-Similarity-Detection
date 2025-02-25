from src.db.db_connect import get_connection



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