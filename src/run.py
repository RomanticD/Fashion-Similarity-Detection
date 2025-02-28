# run.py
from src.app import create_app  # 导入 create_app 函数

app = create_app()  # 创建 Flask 应用实例

if __name__ == "__main__":
    # 启动 Flask 开发服务器
    app.run(debug=True, host='0.0.0.0', port=5001)  # host='0.0.0.0' 表示对外可访问，port=5000 是默认端口