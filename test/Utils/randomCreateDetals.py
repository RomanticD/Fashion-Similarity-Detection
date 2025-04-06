import random
import json

def random_item(i):
    return {
        "image_id": f"{i+1:02d}",  # 从 "01" 到 "400"
        "store_name": random.choice(["怪兽工作室", "潮流天地", "简约风尚"]),
        "brand": random.choice(["unknown", "Nike", "Adidas", "Zara"]),
        "product_name": random.choice(["新款长袖T恤", "韩版牛仔裤", "时尚短裙"]),
        "store_description": random.choice(["优质面料，舒适透气", "适合春夏季穿着", "经典百搭"]),
        "url": "Invalid",
        "rating": round(random.uniform(3.0, 5.0), 1),
        "tags": json.dumps(random.sample(["二次元", "视觉系", "奢侈", "古典", "运动", "休闲"], 3), ensure_ascii=False),
        "sale_status": random.choice(["在售", "售罄"]),
        "size": json.dumps(random.sample(["S", "M", "L", "XL"], random.randint(1, 3)), ensure_ascii=False),
        "waist_type": random.choice(["high", "mid", "low"]),
        "listing_season": random.choice(["2024Spring", "2023Fall", "2025Summer"]),
        "season": random.choice(["spring", "summer", "autumn", "winter"])
    }

sql_lines = []
for i in range(400):
    item = random_item(i)
    line = f"""INSERT INTO image_details (image_id, store_name, brand, product_name, store_description, url, rating, tags, sale_status, size, waist_type, listing_season, season)
VALUES ("{item['image_id']}", "{item['store_name']}", "{item['brand']}", "{item['product_name']}", "{item['store_description']}", "{item['url']}", {item['rating']}, '{item['tags']}', "{item['sale_status']}", '{item['size']}', "{item['waist_type']}", "{item['listing_season']}", "{item['season']}");"""
    sql_lines.append(line)

# 保存为 SQL 文件
with open("insert_products.sql", "w", encoding="utf-8") as f:
    f.write("\n".join(sql_lines))

print("✅ insert_products.sql 文件生成完毕，image_id 从 01 到 400！")