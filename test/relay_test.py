import base64
import unittest
import json
from io import BytesIO

from PIL import Image
from flask import Flask


from src.app.images_relay_route import api_rp  # 导入你的 Blueprint

class ImageRelayTestCase(unittest.TestCase):
    def setUp(self):
        # 创建一个 Flask 测试客户端
        app = Flask(__name__)
        app.register_blueprint(api_rp, url_prefix='/api')  # 注册 Blueprint
        self.client = app.test_client()

    def test_image_relay(self):
        # 创建一个 1x1 的白色图像
        img = Image.new('RGB', (1, 1), color='white')
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        print(img_str)  # 这是你可以在测试中使用的 Base64 字符串
        # 准备一个有效的 Base64 图像字符串（这里应该替换为真实的 Base64 字符串）
        base64_image = img_str  # 替换为有效的 Base64 编码图像数据
        num_return = 5  # 请求返回的相似度最接近的数量

        # 设置请求数据
        data = {
            'num': num_return,
            'image_base64': base64_image,
        }

        # 发送 POST 请求
        response = self.client.post('/api/relay_image', json=data)

        # 验证返回的状态码是 200（成功）
        self.assertEqual(response.status_code, 200, f"Expected status code 200 but got {response.status_code}")

        # 验证返回的 JSON 数据格式
        response_data = json.loads(response.data)
        self.assertIsInstance(response_data, list, "Expected response data to be a list")

        # 验证返回的数据项是否符合预期
        self.assertGreater(len(response_data), 0, "Expected response data to contain at least one item")

        for item in response_data:
            # 验证每个项中是否包含 id, similarity 和 splitted_image_data 字段
            self.assertIn('id', item, "Each item should contain an 'id' field")
            self.assertIn('similarity', item, "Each item should contain a 'similarity' field")

            # 验证相似度是否是数字类型
            self.assertIsInstance(item['similarity'], (float, int), f"Similarity value is not a valid number: {item['similarity']}")
            # 验证 splitted_image_data 是否是字符串类型（Base64 编码字符串）

    """def test_invalid_image(self):
        # 测试无效的图像数据（例如，空字符串或者无效的Base64字符串）
        invalid_base64_image = ""  # 无效的Base64图像数据
        num_return = 5

        # 设置请求数据
        data = {
            'num': num_return,
            'image': invalid_base64_image,
        }

        # 发送 POST 请求
        response = self.client.post('/api/relay_image', json=data)

        # 验证返回的状态码是 400（错误的请求）
        self.assertEqual(response.status_code, 400, f"Expected status code 400 but got {response.status_code}")
        response_data = json.loads(response.data)

        # 验证返回的错误信息是否符合预期
        self.assertIn('error', response_data, "Expected error message in response")
        self.assertEqual(response_data['error'], 'No image data found', "Unexpected error message")
"""
if __name__ == '__main__':
    unittest.main()