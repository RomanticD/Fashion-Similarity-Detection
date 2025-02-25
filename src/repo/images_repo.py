from src.db.db_connect import get_connection


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
