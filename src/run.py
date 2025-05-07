# run.py
import logging
from werkzeug.serving import run_simple
from src.app import create_app  # 导入 create_app 函数
from src.core.vector_index import VectorIndex  # 导入 VectorIndex 类

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 重建索引函数
def rebuild_vector_index():
    """
    Rebuild the vector index before starting the server.
    使用线程安全的方式重建，但在服务器启动时强制重建
    """
    logger.info("Rebuilding vector index before starting the server...")
    try:
        # 创建 VectorIndex 类的实例
        index_manager = VectorIndex()
        # 服务器启动时强制重建索引，忽略锁和时间限制
        result = index_manager.thread_safe_rebuild_index(force=True)
        
        if result:
            logger.info("Vector index successfully rebuilt")
            return True
        else:
            logger.warning("No vectors found in database for index building")
            return False
    except Exception as e:
        logger.error(f"Error rebuilding index: {e}", exc_info=True)
        return False

# 创建 Flask 应用实例
app = create_app()

if __name__ == "__main__":
    try:
        # 首先重建索引
        rebuild_result = rebuild_vector_index()
        if not rebuild_result:
            logger.warning("Vector index rebuild failed or no vectors found. Starting server anyway...")
        
        logger.info("Starting Flask development server...")
        # 使用 Werkzeug 的 run_simple 启动服务器，提供更多的控制选项
        run_simple(
            '0.0.0.0',  # 主机名
            5001,       # 端口
            app,        # 应用实例
            use_reloader=True,  # 启用自动重载
            use_debugger=True,  # 启用调试器
            threaded=True,      # 启用线程
            passthrough_errors=False,  # 不通过错误（让 Flask 处理）
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)