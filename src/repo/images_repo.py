from src.db.db_connect import get_connection


def create_splitted_image_db(splitted_image_id, splitted_image_path, original_image_id, bounding_box=None, vector=None, splitted_image_data=None):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            sql = """
                INSERT INTO splitted_images (
                    splitted_image_id, 
                    splitted_image_path, 
                    original_image_id, 
                    bounding_box, 
                    vector, 
                    splitted_image_data
                )
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (
                splitted_image_id, 
                splitted_image_path, 
                original_image_id, 
                bounding_box, 
                vector, 
                splitted_image_data
            ))
        conn.commit()
        return True
    except Exception as e:
        print("Error creating splitted image record:", e)
        conn.rollback()
        return False
    finally:
        conn.close()


def read_splitted_image_db(splitted_image_id):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            sql = "SELECT * FROM splitted_images WHERE id = %s"
            cursor.execute(sql, (splitted_image_id,))
            result = cursor.fetchone()
        return result
    except Exception as e:
        print("Error reading splitted image record:", e)
        return None
    finally:
        conn.close()


def update_splitted_image_db(splitted_image_id, new_path=None, new_original_id=None, new_box=None, new_vector=None, new_image_data=None):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            # Build the update query dynamically based on provided parameters
            update_parts = []
            params = []
            
            if new_path is not None:
                update_parts.append("splitted_image_path = %s")
                params.append(new_path)
                
            if new_original_id is not None:
                update_parts.append("original_image_id = %s")
                params.append(new_original_id)
                
            if new_box is not None:
                update_parts.append("bounding_box = %s")
                params.append(new_box)
                
            if new_vector is not None:
                update_parts.append("vector = %s")
                params.append(new_vector)
                
            if new_image_data is not None:
                update_parts.append("splitted_image_data = %s")
                params.append(new_image_data)
            
            # If no parameters were provided, return early
            if not update_parts:
                return False
                
            # Complete the parameter list with the ID
            params.append(splitted_image_id)
            
            # Construct and execute the SQL query
            sql = f"""
                UPDATE splitted_images
                SET {', '.join(update_parts)}
                WHERE id = %s
            """
            cursor.execute(sql, tuple(params))
            
        conn.commit()
        return True
    except Exception as e:
        print("Error updating splitted image record:", e)
        conn.rollback()
        return False
    finally:
        conn.close()


def delete_split_image_db(splitted_image_id):
    from src.core.vector_index import VectorIndex  # 导入放在函数内部以避免循环导入

    conn = get_connection()
    success = False
    try:
        with conn.cursor() as cursor:
            sql = "DELETE FROM splitted_images WHERE id = %s"
            cursor.execute(sql, (splitted_image_id,))
        conn.commit()
        success = True
    except Exception as e:
        print("Error deleting splitted_image:", e)
        conn.rollback()
        success = False
    finally:
        conn.close()

    # 如果删除成功，重建向量索引
    if success:
        try:
            # 创建向量索引实例并重建索引
            vector_index = VectorIndex()
            vector_index.rebuild_index()
            print(f"Vector index rebuilt after deleting image {splitted_image_id}")
        except Exception as e:
            print(f"Warning: Failed to rebuild vector index after deleting image: {e}")

    return success


def get_splitted_images_by_original_id(original_image_id):
    """Get all splitted images associated with an original image"""
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            sql = "SELECT * FROM splitted_images WHERE original_image_id = %s"
            cursor.execute(sql, (original_image_id,))
            result = cursor.fetchall()
        return result
    except Exception as e:
        print("Error fetching splitted images:", e)
        return []
    finally:
        conn.close()
