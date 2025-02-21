# app.py
# import sys
# import os
# from pathlib import Path
# from PIL import Image
# import numpy as np
# import streamlit as st
#
# # 导入刚才创建的模块
# from image_processing import split_image_vertically, run_inference, prepare_transform
#
# # 设置 Python 路径
# current_dir = Path(__file__).parent.resolve()
# groundingdino_path = current_dir / 'GroundingDINO'
# sys.path.append(str(groundingdino_path))
#
# # 导入必要的库
# from GroundingDINO.groundingdino.util.inference import load_model
#
# if 'clothes_bboxes' not in st.session_state:
#     st.session_state.clothes_bboxes = []
#
# # Streamlit 应用布局
# st.title('服装款式相似性检测')
#
# # Model Backbone
# CONFIG_PATH = groundingdino_path / 'groundingdino' / 'config' / 'GroundingDINO_SwinT_OGC.py'
# WEIGHTS_PATH = current_dir / 'groundingdino_swint_ogc.pth'
#
# # 检查权重文件是否存在
# if not WEIGHTS_PATH.exists():
#     st.error(f"权重文件未找到，请确保将 '{WEIGHTS_PATH}' 放置在项目根目录中。")
#     st.stop()
#
# # 加载模型
# model = load_model(str(CONFIG_PATH), str(WEIGHTS_PATH), device='cpu')
#
# # 获取变换函数
# transform = prepare_transform()
#
# # 配置参数
# BOX_THRESHOLD = st.sidebar.slider('服装检测灵敏度:', min_value=0.0, max_value=1.0, value=0.3)
# TEXT_PROMPT = st.text_input('Text Prompt:', value="clothes")
#
# # 输入输出配置
# gallery_path = current_dir / 'data'
# st.write(gallery_path)
# Prod_ID = st.text_input('Product ID:', value="id_00008000")
# output_dir = os.path.join(gallery_path, Prod_ID)
# os.makedirs(output_dir, exist_ok=True)
#
# # 文件上传
# upload_img_file = st.file_uploader('选择图像', type=['jpg', 'jpeg', 'png'])
# FRAME_WINDOW = st.image([])
#
# clothes_bboxes = []
#
# if upload_img_file is not None:
#     img = Image.open(upload_img_file).convert("RGB")
#     image = np.asarray(img)
#     image_transformed, _ = transform(img, None)
#
#     # 设定分段长度为图像宽度的3倍
#     segment_height = image.shape[1] * 3
#     segments = split_image_vertically(image, segment_height)
#     FRAME_WINDOW.image(img, channels='BGR')
#
# # 触发:服装检测
# if st.button('检测页面中的服装'):
#     try:
#         st.session_state.clothes_bboxes = run_inference(model, transform, segments, TEXT_PROMPT, BOX_THRESHOLD, FRAME_WINDOW)
#         # 在结果图下方显示所有返回的 clothes_bboxes 图像
#         st.write("检测到的服装区域：")
#         for idx, bbox in enumerate(st.session_state.clothes_bboxes):
#             st.image(bbox, caption=f"服装 {idx + 1}")
#     except Exception as e:
#         st.error(f"检测过程中出错: {e}")
#
# # 添加按钮以触发相似度检测
# if st.button('检测服装相似度'):
#     if not st.session_state.clothes_bboxes:
#         st.write("请先点击‘检测页面中的服装’按钮进行检测。")
#     else:
#         st.write("潜在的雷同款式：")
#         for idx, bbox in enumerate(st.session_state.clothes_bboxes):
#             # 将 bbox 保存为临时图像文件
#             bbox_image = Image.fromarray(bbox)
#             tmp_image_path = './img_tmp/intermd.jpg'
#             os.makedirs('./img_tmp', exist_ok=True)
#             bbox_image.save(tmp_image_path)
#
#             # 调用 mmfashion_retrieval.py 进行检测
#             os.system(f"python3 mmfashion/tools/test_retriever.py --input {tmp_image_path}")
#
#             # 显示检测结果图像
#             output_image_path = './output.png'
#             if os.path.exists(output_image_path):
#                 st.image(output_image_path, caption=f"服装 {idx + 1} 相似款", use_column_width=True)
#             else:
#                 st.write(f"未能生成服装 {idx + 1} 的相似款图像。")

import sys
import os
from pathlib import Path
from PIL import Image
import numpy as np
import streamlit as st
# 从 groundingdino_handler 模块导入 detect_clothes_in_image 函数
from groundingdino_handler import detect_clothes_in_image

if 'clothes_bboxes' not in st.session_state:
    st.session_state.clothes_bboxes = []

# Streamlit 应用布局
st.title('服装款式相似性检测')

# 输入输出配置
current_dir = Path(__file__).parent.resolve()
gallery_path = current_dir / 'data'
st.write(gallery_path)
Prod_ID = st.text_input('Product ID:', value="id_00008000")
output_dir = os.path.join(gallery_path, Prod_ID)
os.makedirs(output_dir, exist_ok=True)

# 文件上传
upload_img_file = st.file_uploader('选择图像', type=['jpg', 'jpeg', 'png'])
FRAME_WINDOW = st.image([])

clothes_bboxes = []

if upload_img_file is not None:
    img = Image.open(upload_img_file).convert("RGB")
    image = np.asarray(img)
    try:
        FRAME_WINDOW.image(img, channels='BGR')
    except Exception as e:
        st.error(f"图片保存失败: {e}")

# 触发:服装检测
if st.button('检测页面中的服装'):
    if upload_img_file is None:
        st.write("请先上传图像。")
    else:
        try:
            st.session_state.clothes_bboxes = detect_clothes_in_image(image, FRAME_WINDOW)
            st.write("检测到的服装区域：")
            for idx, bbox in enumerate(st.session_state.clothes_bboxes):
                st.image(bbox, caption=f"服装 {idx + 1}")
        except Exception as e:
            st.error(f"检测过程中出错1: {e}")

# 添加按钮以触发相似度检测
if st.button('检测服装相似度'):
    if not st.session_state.clothes_bboxes:
        st.write("请先点击‘检测页面中的服装’按钮进行检测。")
    else:
        st.write("潜在的雷同款式：")
        for idx, bbox in enumerate(st.session_state.clothes_bboxes):
            # 将 bbox 保存为临时图像文件
            bbox_image = Image.fromarray(bbox)
            tmp_image_path = './img_tmp/intermd.jpg'
            os.makedirs('./img_tmp', exist_ok=True)
            bbox_image.save(tmp_image_path)

            # 调用 mmfashion_retrieval.py 进行检测
            os.system(f"python3 mmfashion/tools/test_retriever.py --input {tmp_image_path}")

            # 显示检测结果图像
            output_image_path = './output.png'
            if os.path.exists(output_image_path):
                st.image(output_image_path, caption=f"服装 {idx + 1} 相似款", use_column_width=True)
            else:
                st.write(f"未能生成服装 {idx + 1} 的相似款图像。")

