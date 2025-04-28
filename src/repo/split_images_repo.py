from flask import current_app
from src.db.db_connect import get_connection
import logging
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add a database connection lock to prevent issues with concurrent access
_db_lock = threading.Lock()


def create_split_image_db(splitted_image_id, splitted_image_path, original_image_id, bounding_box=None, vector=None, splitted_image_data=None):
    """
    向split_images表插入一条记录
    """
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            sql = """
                INSERT INTO split_images (
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
        logger.error(f"Error creating split image record: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()


def read_split_image_db(splitted_image_id):
    """
    从split_images表读取一条记录
    """
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            sql = "SELECT * FROM split_images WHERE splitted_image_id = %s"
            cursor.execute(sql, (splitted_image_id,))
            result = cursor.fetchone()
        return result
    except Exception as e:
        logger.error(f"Error reading split image record: {e}")
        return None
    finally:
        conn.close()


def update_split_image_db(splitted_image_id, new_path=None, new_original_id=None, new_box=None, new_vector=None, new_image_data=None):
    """
    更新split_images表中的一条记录
    """
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
                UPDATE split_images
                SET {', '.join(update_parts)}
                WHERE splitted_image_id = %s
            """
            cursor.execute(sql, tuple(params))
            
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Error updating split image record: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()


def delete_split_image_db(splitted_image_id):
    """
    从split_images表删除一条记录
    """
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            sql = "DELETE FROM split_images WHERE splitted_image_id = %s"
            cursor.execute(sql, (splitted_image_id,))
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Error deleting split image record: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()


def save_to_db(splitted_image_id, splitted_image_path, original_image_id, bounding_box, binary_data, vector):
    """
    保存分割后的图片数据到split_images表
    
    Args:
        splitted_image_id (str): 分割图像ID
        splitted_image_path (str): 分割图像路径
        original_image_id (str): 原始图像ID
        bounding_box (str): 边界框信息
        binary_data (bytes): 图像二进制数据
        vector (str): 特征向量JSON字符串
        
    Returns:
        bool: 操作是否成功
    """
    # Use connection lock to prevent concurrent database issues
    with _db_lock:
        # 连接到数据库
        conn = get_connection()
        cursor = conn.cursor()

        try:
            logger.info(f"正在保存分割图像到数据库: {splitted_image_id}, 原图ID: {original_image_id}")
            
            # 检查必要参数
            if not splitted_image_id or not binary_data or not vector:
                logger.error(f"缺少必要参数: ID={splitted_image_id}, 有二进制数据={bool(binary_data)}, 有特征向量={bool(vector)}")
                return False
                
            # 检查该ID是否已存在（防止重复）
            cursor.execute("SELECT COUNT(*) FROM split_images WHERE splitted_image_id = %s", (splitted_image_id,))
            if cursor.fetchone()[0] > 0:
                logger.warning(f"分割图像ID已存在，将更新记录: {splitted_image_id}")
                
                # 更新现有记录
                cursor.execute("""
                    UPDATE split_images 
                    SET splitted_image_path = %s, 
                        original_image_id = %s, 
                        bounding_box = %s, 
                        splitted_image_data = %s, 
                        vector = %s
                    WHERE splitted_image_id = %s
                """, (splitted_image_path, original_image_id, bounding_box, binary_data, vector, splitted_image_id))
            else:
                # 插入图像数据和特征到数据库
                cursor.execute("""
                    INSERT INTO split_images (splitted_image_id, splitted_image_path, original_image_id, bounding_box, splitted_image_data, vector)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (splitted_image_id, splitted_image_path, original_image_id, bounding_box, binary_data, vector))

            # 提交事务
            conn.commit()
            logger.info(f"图像 {splitted_image_id} 成功保存到split_images表，包含特征向量")
            return True
        except Exception as e:
            logger.error(f"保存图像到split_images表时出错: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            conn.rollback()
            return False
        finally:
            # 关闭数据库连接
            cursor.close()
            conn.close()


def select_all_vectors():
    """
    获取split_images表中所有图片的向量特征
    """
    with _db_lock:
        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT id, vector FROM split_images")
                rows = cursor.fetchall()
                # 如果没有查询到数据，返回空列表
                if not rows:
                    logger.info("No data found in the 'split_images' table.")
                    return []

                return rows
        except Exception as e:
            logger.error(f"Error during select: {e}")
            conn.rollback()  # 回滚事务
            return []  # 发生异常时返回空列表而不是 False
        finally:
            conn.close()  # 确保连接被关闭


def select_image_data_by_id(id):
    """
    通过ID获取split_images表中的图片数据
    """
    with _db_lock:
        conn = get_connection()
        if not conn:
            logger.error("Failed to connect to the database")
            return None

        try:
            with conn.cursor() as cursor:
                sql = "SELECT splitted_image_data FROM split_images WHERE id = %s"
                logger.debug(f"Executing SQL: {sql} with id={id}")
                cursor.execute(sql, (id,))
                row = cursor.fetchone()

                if not row:
                    logger.warning(f"No record found for id={id}")
                    return None
                return row

        except Exception as e:
            logger.error(f"Error selecting image data by id {id}: {str(e)}", exc_info=True)
            conn.rollback()
            return None
        finally:
            conn.close()


def select_multiple_image_data_by_ids(ids):
    """
    通过ID列表获取split_images表中的多张图片数据
    
    Parameters:
    ids (list): List of image IDs to fetch

    Returns:
    dict: Dictionary mapping ID to image data
    """
    if not ids:
        return {}

    with _db_lock:
        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                # Use a more efficient approach - fetch in batches to avoid overwhelming the database
                result = {}
                batch_size = 50  # Adjust based on your database capacity

                # Process in batches
                for i in range(0, len(ids), batch_size):
                    batch_ids = ids[i:i + batch_size]
                    placeholders = ', '.join(['%s'] * len(batch_ids))
                    sql = f"SELECT id, splitted_image_data FROM split_images WHERE id IN ({placeholders})"
                    cursor.execute(sql, batch_ids)
                    rows = cursor.fetchall()

                    # Add to result dictionary
                    for row in rows:
                        result[row['id']] = row

                return result
        except Exception as e:
            logger.error(f"Error fetching multiple image data: {e}")
            return {}
        finally:
            conn.close()


def get_split_images_by_original_id(original_image_id):
    """
    获取与原图相关联的所有分割图片记录
    """
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            sql = "SELECT * FROM split_images WHERE original_image_id = %s"
            cursor.execute(sql, (original_image_id,))
            result = cursor.fetchall()
        return result
    except Exception as e:
        logger.error(f"Error fetching split images: {e}")
        return []
    finally:
        conn.close()
