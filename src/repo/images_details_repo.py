from src.db.db_connect import get_connection

def read_image_detail (original_image_id):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            sql = "SELECT * FROM image_details WHERE image_id = %s"
            cursor.execute(sql, (original_image_id,))
            result = cursor.fetchone()
        return result
    except Exception as e:
        print("Error reading image_detail:", e)
        return None
    finally:
        conn.close()
