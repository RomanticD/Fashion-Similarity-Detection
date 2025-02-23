from src.app import create_app  # 修改导入路径

app = create_app()

if __name__ == "__main__":
    # 可根据需要修改 host / port
    app.run(host="0.0.0.0", port=5001, debug=True)