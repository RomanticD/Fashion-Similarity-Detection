import sqlite3
from pathlib import Path

def clear_splitted_images(db_path: str = "../data/fashion.db"):
    """
    清空splitted_images表中的所有数据
    Args:
        db_path (str): 数据库文件路径，默认使用项目数据目录下的fashion.db
    """
    db_abs_path = str(Path(__file__).parent.parent / db_path)
    
    try:
        conn = sqlite3.connect(db_abs_path)
        cursor = conn.cursor()
        
        # 执行删除操作
        cursor.execute("DELETE FROM splitted_images")
        conn.commit()
        
        print(f"成功清空splitted_images表，共删除{cursor.rowcount}条记录")
        
    except sqlite3.Error as e:
        print(f"数据库操作失败: {str(e)}")
        conn.rollback()
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    clear_splitted_images()