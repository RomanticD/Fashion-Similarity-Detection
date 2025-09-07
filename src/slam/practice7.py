import cv2
import numpy as np
import jaxlie  # 替代Sophus的李代数功能
import open3d as o3d


def find_orb_features(image, nfeatures=500):
    """模块1：提取ORB特征点与描述子"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(
        nfeatures=nfeatures,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31
    )
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors


def match_features(descriptors1, descriptors2, ratio=0.75):
    """模块2：特征匹配与误匹配筛选"""
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = []
    for m, n in raw_matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    return good_matches


def pose_estimation_2d2d(keypoints1, keypoints2, matches, K):
    """模块3：2D-2D对极约束估计相机运动"""
    points1 = [keypoints1[m.queryIdx].pt for m in matches]
    points2 = [keypoints2[m.trainIdx].pt for m in matches]
    points1 = np.array(points1, dtype=np.float64)
    points2 = np.array(points2, dtype=np.float64)

    cx, cy = K[0, 2], K[1, 2]
    fx, fy = K[0, 0], K[1, 1]
    essential_matrix, mask = cv2.findEssentialMat(
        points1, points2,
        focal=fx, pp=(cx, cy),
        method=cv2.RANSAC,
        prob=0.999, threshold=1.0
    )

    _, R, t, mask = cv2.recoverPose(
        essential_matrix, points1, points2,
        focal=fx, pp=(cx, cy), mask=mask
    )

    good_matches = [matches[i] for i in range(len(matches)) if mask[i] == 1]
    return R, t, good_matches


def triangulation(keypoints1, keypoints2, matches, R, t, K):
    """模块4：三角测量估计3D空间点"""
    T1 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0]], dtype=np.float64)
    T2 = np.hstack((R, t))

    points1_pixel = [keypoints1[m.queryIdx].pt for m in matches]
    points2_pixel = [keypoints2[m.trainIdx].pt for m in matches]
    points1_pixel = np.array(points1_pixel, dtype=np.float64)
    points2_pixel = np.array(points2_pixel, dtype=np.float64)

    def pixel2normal(pixel, K):
        cx, cy = K[0, 2], K[1, 2]
        fx, fy = K[0, 0], K[1, 1]
        x = (pixel[0] - cx) / fx
        y = (pixel[1] - cy) / fy
        return np.array([x, y], dtype=np.float64)

    points1_normal = [pixel2normal(p, K) for p in points1_pixel]
    points2_normal = [pixel2normal(p, K) for p in points2_pixel]
    points1_normal = np.array(points1_normal, dtype=np.float64)
    points2_normal = np.array(points2_normal, dtype=np.float64)

    points_4d = cv2.triangulatePoints(
        T1, T2,
        points1_normal.T,
        points2_normal.T
    )

    points_3d = points_4d[:3, :] / points_4d[3, :]
    points_3d = points_3d.T
    return points_3d


def bundle_adjustment_gauss_newton(points_3d, points_2d, K, pose):
    """模块5：BA优化（jaxlie适配）"""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    iterations = 10
    n_points = len(points_3d)

    for iter in range(iterations):
        H = np.zeros((6, 6), dtype=np.float64)
        b = np.zeros((6, 1), dtype=np.float64)
        total_cost = 0

        for i in range(n_points):
            P = points_3d[i].reshape(3, 1)  # 世界系3D点（3x1）
            u_obs = points_2d[i][0]
            v_obs = points_2d[i][1]

            # jaxlie.SE3的乘法（世界系→相机系）
            P_world = P.flatten()  # 3x1 → 3维向量
            P_cam = pose @ jaxlie.SE3.from_translation(P_world)
            X, Y, Z = P_cam.translation()  # 修正：调用translation()方法
            inv_Z = 1.0 / Z if Z != 0 else 1e-6
            inv_Z2 = inv_Z ** 2

            # 投影坐标与残差计算
            u_proj = fx * X * inv_Z + cx
            v_proj = fy * Y * inv_Z + cy
            e = np.array([[u_obs - u_proj], [v_obs - v_proj]], dtype=np.float64)
            total_cost += np.sum(e ** 2)

            # 雅可比矩阵计算
            J = np.zeros((2, 6), dtype=np.float64)
            J[0, 0] = -fx * inv_Z
            J[0, 1] = 0
            J[0, 2] = fx * X * inv_Z2
            J[0, 3] = fx * X * Y * inv_Z2
            J[0, 4] = -fx - fx * (X ** 2) * inv_Z2
            J[0, 5] = fx * Y * inv_Z

            J[1, 0] = 0
            J[1, 1] = -fy * inv_Z
            J[1, 2] = fy * Y * inv_Z2
            J[1, 3] = fy + fy * (Y ** 2) * inv_Z2
            J[1, 4] = -fy * X * Y * inv_Z2
            J[1, 5] = -fy * X * inv_Z

            # 累积Hessian和残差向量
            H += J.T @ J
            b += -J.T @ e

        # 求解增量方程（Cholesky分解）
        try:
            L = np.linalg.cholesky(H)
            delta_x = np.linalg.solve(L, b)
            delta_x = np.linalg.solve(L.T, delta_x)
        except np.linalg.LinAlgError:
            print(f"迭代{iter}：Hessian矩阵奇异，停止优化")
            break

        #  更新位姿（李代数指数映射+左乘）
        delta_x = delta_x.flatten()
        delta_se3 = jaxlie.SE3.exp(delta_x)
        pose = delta_se3 @ pose

        # 打印迭代信息
        print(f"BA迭代{iter+1}/{iterations}：总代价={total_cost:.4f}，增量范数={np.linalg.norm(delta_x):.6f}")

        # 提前收敛判断
        if np.linalg.norm(delta_x) < 1e-6:
            print("增量足够小，提前收敛")
            break

    return pose


def main():
    # -------------------------- 1. 配置参数与路径 --------------------------
    img1_path = "/Users/sunyuliang/Desktop/AppBuilder/Python/RD-Test/Assets/rgb/1305031452.791720.png"
    img2_path = "/Users/sunyuliang/Desktop/AppBuilder/Python/RD-Test/Assets/rgb/1305031452.859642.png"
    K = np.array([[520.9, 0, 325.1],
                  [0, 521.0, 249.7],
                  [0, 0, 1]], dtype=np.float64)

    # -------------------------- 2. 读取图像 --------------------------
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    assert img1 is not None and img2 is not None, "请检查图像路径是否正确！"

    # -------------------------- 3. 提取ORB特征与匹配 --------------------------
    print("步骤1：提取ORB特征与匹配...")
    keypoints1, descriptors1 = find_orb_features(img1, nfeatures=500)
    keypoints2, descriptors2 = find_orb_features(img2, nfeatures=500)
    good_matches = match_features(descriptors1, descriptors2, ratio=0.75)
    print(f"原始匹配对数量：{len(descriptors1)}，筛选后匹配对数量：{len(good_matches)}")

    # 绘制匹配结果
    img_matches = cv2.drawMatches(
        img1, keypoints1, img2, keypoints2,
        good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imshow("ORB Feature Matches", img_matches)
    cv2.waitKey(0)

    # -------------------------- 4. 2D-2D对极约束求相机运动 --------------------------
    print("\n步骤2：2D-2D对极约束估计R和t...")
    R_2d2d, t_2d2d, good_matches_filtered = pose_estimation_2d2d(
        keypoints1, keypoints2, good_matches, K
    )
    print(f"本质矩阵筛选后匹配对数量：{len(good_matches_filtered)}")
    print("旋转矩阵R:\n", R_2d2d)
    print("平移向量t（单位：米，单目有尺度不确定性）:\n", t_2d2d)

    # -------------------------- 5. 三角测量估计3D点 --------------------------
    print("\n步骤3：三角测量估计3D点...")
    points_3d = triangulation(
        keypoints1, keypoints2, good_matches_filtered,
        R_2d2d, t_2d2d, K
    )
    print(f"三角化得到{len(points_3d)}个3D点")

    # 可视化3D点
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    colors = []
    for m in good_matches_filtered:
        u, v = keypoints1[m.queryIdx].pt
        u, v = int(u), int(v)
        b, g, r = img1[v, u]
        colors.append([r/255.0, g/255.0, b/255.0])
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd], window_name="Triangulated 3D Points")

    # -------------------------- 6. 3D-2D PnP求解位姿 --------------------------
    print("\n步骤4：3D-2D PnP求解位姿...")
    points_2d_pnp = [keypoints2[m.trainIdx].pt for m in good_matches_filtered]
    points_2d_pnp = np.array(points_2d_pnp, dtype=np.float64)

    # 6.1 OpenCV EPnP求解
    retval, r_vec, t_vec, inliers = cv2.solvePnPRansac(
        points_3d, points_2d_pnp, K, distCoeffs=None,
        flags=cv2.SOLVEPNP_EPNP
    )
    R_pnp, _ = cv2.Rodrigues(r_vec)
    t_pnp = t_vec.squeeze().reshape(3,).astype(np.float64)  # 强制float64
    print("OpenCV EPnP估计的R:\n", R_pnp)
    print("OpenCV EPnP估计的t（shape=(3,)）:\n", t_pnp)

    # 6.2 用齐次矩阵创建初始SE3位姿（避免广播问题）
    print("\n步骤5：BA优化PnP位姿...")
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_pnp
    T[:3, 3] = t_pnp
    pose_initial = jaxlie.SE3.from_matrix(T)

    # BA优化
    pose_optimized = bundle_adjustment_gauss_newton(
        points_3d, points_2d_pnp, K, pose_initial
    )

    # -------------------------- 提取优化后的R和t --------------------------
    # 方案：先将SE3转为4x4齐次矩阵，再从矩阵中提取R和t
    T_optimized = pose_optimized.as_matrix()  # SE3→4x4齐次矩阵
    R_optimized = T_optimized[:3, :3]        # 提取3x3旋转矩阵
    t_optimized = T_optimized[:3, 3].reshape(3, 1)  # 提取3x1平移向量

    # 打印优化结果
    print("\nBA优化后的R:\n", R_optimized)
    print("BA优化后的t:\n", t_optimized)

    # -------------------------- 7. 释放资源 --------------------------
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
