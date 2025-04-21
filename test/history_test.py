# test/history_test.py
from src.app import create_app

app = create_app()

import requests

def run_test():
    for rule in app.url_map.iter_rules():
        print(rule)
    search_id = "01"
    url = f"http://localhost:5001/search_history/{search_id}"

    response = requests.get(url)

    print("状态码:", response.status_code)
    try:
            print("返回数据:", response.json())
    except Exception:
             print("不是 JSON 格式:", response.text)

if __name__ == '__main__':
    run_test()