import os
from pathlib import Path
from PIL import Image
import numpy as np
import streamlit as st
# 从 groundingdino_handler 模块导入 detect_clothes_in_image 函数
from src.core.groundingdino_handler import detect_clothes_in_image

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

