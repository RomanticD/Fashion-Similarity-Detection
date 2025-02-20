from src.flask import app

if __name__ == "__main__":
    # 可根据需要修改 host / port
    app.run(host="0.0.0.0", port=5001, debug=True)