import pymysql

from src.db.db_connect import get_connection


def insert_search_history(data):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            sql = """
                INSERT INTO image_history (
                    search_id, search_image_id,
                    result_image_id_1, result_image_id_2, result_image_id_3,
                    similarity_1, similarity_2, similarity_3
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (
                data['search_id'],
                data['search_image_id'],
                data.get('result_image_id_1'),
                data.get('result_image_id_2'),
                data.get('result_image_id_3'),
                data.get('similarity_1'),
                data.get('similarity_2'),
                data.get('similarity_3')
            ))
        conn.commit()
        return True
    except Exception as e:
        print("Error inserting into search_history:", e)
        return False
    finally:
        conn.close()

def read_search_history(search_id):
    conn = get_connection()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            sql = "SELECT * FROM image_history WHERE search_id = %s"
            cursor.execute(sql, (search_id,))
            result = cursor.fetchone()
        return result
    except Exception as e:
        print("Error reading search_history:", e)
        return None
    finally:
        conn.close()
