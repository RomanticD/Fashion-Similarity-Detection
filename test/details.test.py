from src.repo.images_details_repo import read_image_detail

if __name__ == "__main__":
    image_id = 1  # 这里填你要查询的 ID
    result = read_image_detail(image_id)
    print("查询结果:", result)