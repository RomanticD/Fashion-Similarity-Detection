# run.py
import logging
from werkzeug.serving import run_simple
from src.app import create_app  # 导入 create_app 函数

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建 Flask 应用实例
app = create_app()

if __name__ == "__main__":
    try:
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