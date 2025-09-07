import cv2
import numpy as np
import open3d as o3d


def image_undistort(image, K, dist_coeffs):
    """
    任务1：实现图像去畸变
    输入：畸变图像（彩色图为3通道，深度图为单通道）、相机内参K、畸变系数dist_coeffs
    输出：无畸变图像（裁剪黑边后）
    """
    # 1. 获取图像尺寸（高度h、宽度w）
    h, w = image.shape[:2]
    # 2. 计算去畸变后的最优内参（保留所有像素，避免信息丢失）
    # 参数1：原始内参K；参数2：畸变系数；参数3：图像尺寸；参数4：alpha=1表示保留所有像素；参数5：输出图像尺寸
    new_K, roi = cv2.getOptimalNewCameraMatrix(
        cameraMatrix=K,
        distCoeffs=dist_coeffs,
        imageSize=(w, h),
        alpha=1,
        newImgSize=(w, h)
    )
    # 3. 执行去畸变（核心函数）
    undistorted_img = cv2.undistort(
        src=image,
        cameraMatrix=K,
        distCoeffs=dist_coeffs,
        newCameraMatrix=new_K
    )
    # 4. 裁剪图像黑边（roi为有效区域的坐标：x=左边界，y=上边界，w_roi=宽度，h_roi=高度）
    x, y, w_roi, h_roi = roi
    undistorted_img = undistorted_img[y:y+h_roi, x:x+w_roi]
    return undistorted_img


def rgbd_to_pointcloud(color_img, depth_img, K, depth_scale=1000.0, max_depth=10.0):
    """
    任务2：从RGB-D图像生成3D点云
    输入：
        color_img：去畸变后的彩色图（BGR格式）
        depth_img：去畸变后的深度图（16位无符号整数）
        K：相机内参（去畸变后的内参，需与去畸变图像匹配）
        depth_scale：深度缩放因子（TUM数据集默认1000，即像素值=实际深度*1000）
        max_depth：最大有效深度（过滤过远的噪声点，默认10米）
    输出：Open3D格式的带颜色3D点云
    """
    # 1. 提取相机内参关键参数（fx=焦距x，fy=焦距y，cx=光心x，cy=光心y）
    fx, fy = K[0][0], K[1][1]
    cx, cy = K[0][2], K[1][2]
    # 2. 获取深度图尺寸（确保与彩色图尺寸一致）
    h_depth, w_depth = depth_img.shape
    h_color, w_color = color_img.shape[:2]
    assert (h_depth == h_color) and (w_depth == w_color), "彩色图与深度图尺寸不匹配！"

    # 3. 初始化点云（3D坐标）和颜色（RGB）容器
    points = []  # 存储格式：[X, Y, Z]（单位：米）
    colors = []  # 存储格式：[R, G, B]（单位：0~1，Open3D要求）

    # 4. 遍历每个像素，按针孔模型计算3D坐标（核心逻辑）
    for v in range(h_depth):  # v：图像行（对应相机坐标系Y方向）
        for u in range(w_depth):  # u：图像列（对应相机坐标系X方向）
            # 4.1 读取深度值（跳过无效深度：0或超过max_depth）
            depth_val = depth_img[v, u]  # 深度图像素值（16位整数）
            if depth_val == 0:
                continue  # 无深度数据（无效点）
            # 转换为实际深度（米），并过滤过远点
            Z = depth_val / depth_scale
            if Z > max_depth:
                continue

            # 4.2 用针孔模型计算3D点坐标（第五讲核心公式）
            # X = (u - cx) * Z / fx （像素u到相机X的转换）
            # Y = (v - cy) * Z / fy （像素v到相机Y的转换）
            # Z = 实际深度（从深度图获取）
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy

            # 4.3 读取彩色图对应像素的颜色（OpenCV读入为BGR，需转为RGB）
            b, g, r = color_img[v, u]  # 彩色图像素：B通道，G通道，R通道
            color_rgb = [r / 255.0, g / 255.0, b / 255.0]  # 归一化到0~1（Open3D要求）

            # 4.4 将3D点和颜色添加到容器
            points.append([X, Y, Z])
            colors.append(color_rgb)

    # 5. 转换为Open3D点云格式（核心数据结构）
    pcd = o3d.geometry.PointCloud()
    # 转换3D点为numpy数组，再转为Open3D向量
    pcd.points = o3d.utility.Vector3dVector(np.array(points, dtype=np.float64))
    # 转换颜色为numpy数组，再转为Open3D向量
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors, dtype=np.float64))
    return pcd


def main():
    # -------------------------- 1. 数据准备与读取 --------------------------
    # 图像路径
    color_path = "/Users/sunyuliang/Desktop/AppBuilder/Python/RD-Test/Assets/rgb/1305031452.791720.png"
    depth_path = "/Users/sunyuliang/Desktop/AppBuilder/Python/RD-Test/Assets/depth/1305031453.374112.png"
    # 读取彩色图（cv2.imread默认读取为BGR格式，符合后续颜色转换逻辑）
    color_img = cv2.imread(color_path)
    # 读取深度图（cv2.IMREAD_UNCHANGED：保留16位原始数据，不转为8位）
    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    # 检查图像是否成功读取（避免路径错误导致崩溃）
    assert color_img is not None, f"彩色图读取失败！请检查路径：{color_path}"
    assert depth_img is not None, f"深度图读取失败！请检查路径：{depth_path}"
    assert depth_img.dtype == np.uint16, "深度图需为16位无符号整数格式（TUM数据集标准）"

    # -------------------------- 2. 相机参数配置（TUM数据集标准参数） --------------------------
    # 相机内参K（单位：像素）
    K = np.array([
        [520.9, 0.0, 325.1],  # 第一行：fx, 0, cx
        [0.0, 521.0, 249.7],  # 第二行：0, fy, cy
        [0.0, 0.0, 1.0]       # 第三行：0, 0, 1
    ], dtype=np.float32)
    # 畸变系数（5个参数，“径向+切向畸变”：k1, k2, p1, p2, k3）
    dist_coeffs = np.array([
        [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05, -0.00434008]
    ], dtype=np.float32)

    # -------------------------- 3. 图像去畸变 --------------------------
    # 彩色图去畸变（3通道）
    undistorted_color = image_undistort(color_img, K, dist_coeffs)
    # 深度图去畸变（单通道16位）
    undistorted_depth = image_undistort(depth_img, K, dist_coeffs)
    # 注意：去畸变后内参需更新（与去畸变图像匹配），通过getOptimalNewCameraMatrix重新计算
    # 重新计算去畸变后的内参（基于去畸变后的图像尺寸）
    h_undistort, w_undistort = undistorted_color.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(
        K, dist_coeffs, (w_undistort, h_undistort), alpha=1, newImgSize=(w_undistort, h_undistort)
    )

    # -------------------------- 4. 生成3D点云 --------------------------
    # 调用函数生成点云（depth_scale=1000，max_depth=10米过滤噪声）
    pcd = rgbd_to_pointcloud(
        color_img=undistorted_color,
        depth_img=undistorted_depth,
        K=new_K,  # 必须用去畸变后的内参！
        depth_scale=1000.0,
        max_depth=10.0
    )

    # -------------------------- 5. 结果可视化（验证正确性） --------------------------
    # 显示去畸变后的彩色图（验证畸变修正效果：边缘无弯曲）
    cv2.imshow("Undistorted Color Image", undistorted_color)
    print("提示：按下任意键关闭图像窗口，继续显示点云...")
    cv2.waitKey(0)  # 等待任意键关闭图像窗口
    cv2.destroyAllWindows()  # 释放OpenCV窗口资源

    # 显示3D点云（验证3D结构：颜色与场景对应，轮廓清晰）
    print("提示：点云窗口中，鼠标可拖动旋转视角，滚轮缩放，按下'Q'退出...")
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="RGB-D Point Cloud (From TUM Dataset)",
        width=1280,
        height=720,
        left=50,
        top=50,
        point_show_normal=False  # 关闭法向量显示（加快渲染）
    )


if __name__ == "__main__":
    main()
