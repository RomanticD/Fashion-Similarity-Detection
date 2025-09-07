import cv2
import numpy as np

# 读取RGB图和深度图（假设已对应）
rgb = cv2.imread("rgb.png")
depth = cv2.imread("depth.png", cv2.IMREAD_UNCHANGED)
h, w = rgb.shape[:2]

# 相机内参（假设已校准）
K = np.array([[520.9, 0, 325.1], [0, 521.0, 249.7], [0, 0, 1]], dtype=np.float32)
fx, fy = K[0,0], K[1,1]
cx, cy = K[0,2], K[1,2]

# 取一个特征点（如墙角，假设坐标(200, 150)）
u, v = 200, 150
Z = depth[v, u] / 1000.0  # 假设深度缩放因子1000

# 计算3D坐标
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy

# 投影回2D坐标（验证）
u_proj = int(X * fx / Z + cx)
v_proj = int(Y * fy / Z + cy)

# 若输出接近（如(200,150) vs (200,150)），说明对应
print(f"原始坐标: ({u}, {v}), 投影后坐标: ({u_proj}, {v_proj})")
