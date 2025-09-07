import open3d as o3d
import numpy as np

# # 测试geometry模块（创建PointCloud）
# pcd = o3d.geometry.PointCloud()
# print("✅ geometry.PointCloud 可用")
#
# # 测试utility模块（创建Vector3dVector）
# test_points = np.array([[0,0,0], [1,1,1]], dtype=np.float64)
# pcd.points = o3d.utility.Vector3dVector(test_points)
# print("✅ utility.Vector3dVector 可用")
#
# # 测试可视化（可选）
# o3d.visualization.draw_geometries([pcd], window_name="Test Open3D 0.19.0")
# print("✅ 可视化功能正常")

import sys
print(sys.executable)  # 输出当前Python解释器的路径
