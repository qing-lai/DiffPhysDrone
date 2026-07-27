import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 读取JSON文件（假设文件名为 traj_history.json）
with open('/home/liujiantao/DiffPhysDrone_JT/DiffPhysDrone/jt_high_speed_flight/exps_6.0/20260727_171821/traj_history.json', 'r') as f:
    data = json.load(f)

# 提取无人机数据
points = data['drone_1']
x = [p[0] for p in points]
y = [p[1] for p in points]
z = [p[2] for p in points]

print(f"轨迹点数量: {len(points)}")
print(f"X范围: {min(x):.2f} ~ {max(x):.2f}")
print(f"Y范围: {min(y):.2f} ~ {max(y):.2f}")
print(f"Z范围: {min(z):.2f} ~ {max(z):.2f}")

# 创建三维图
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# 用颜色渐变表示时间（从起点到终点）
colors = np.linspace(0, 1, len(x))
scatter = ax.scatter(x, y, z, c=colors, cmap='jet', s=2, alpha=0.6)
ax.plot(x, y, z, 'k-', linewidth=0.5, alpha=0.2)

# 标注起点和终点
ax.scatter(x[0], y[0], z[0], c='green', s=100, label='起点', marker='o', edgecolors='black', linewidth=2)
ax.scatter(x[-1], y[-1], z[-1], c='red', s=100, label='终点', marker='s', edgecolors='black', linewidth=2)

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('无人机三维飞行轨迹 (颜色表示时间顺序)')

# 添加颜色条
cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
cbar.set_label('时间进度')

ax.legend()

# 调整视角
ax.view_init(elev=20, azim=-60)

plt.tight_layout()
plt.show()

# 绘制俯视图
fig2, ax2 = plt.subplots(figsize=(12, 10))
scatter2 = ax2.scatter(x, y, c=colors, cmap='jet', s=2, alpha=0.6)
ax2.plot(x, y, 'k-', linewidth=0.5, alpha=0.2)
ax2.scatter(x[0], y[0], c='green', s=100, label='起点', marker='o', edgecolors='black', linewidth=2)
ax2.scatter(x[-1], y[-1], c='red', s=100, label='终点', marker='s', edgecolors='black', linewidth=2)
ax2.set_xlabel('X (m)')
ax2.set_ylabel('Y (m)')
ax2.set_title('无人机轨迹俯视图 (XY平面)')
ax2.set_aspect('equal')
ax2.legend()
cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.6)
cbar2.set_label('时间进度')
plt.tight_layout()
plt.show()

# 绘制高度变化图
fig3, ax3 = plt.subplots(figsize=(12, 6))
ax3.plot(range(len(z)), z, 'b-', linewidth=1)
ax3.scatter(0, z[0], c='green', s=80, label='起点', zorder=5)
ax3.scatter(len(z)-1, z[-1], c='red', s=80, label='终点', zorder=5)
ax3.set_xlabel('轨迹点序号')
ax3.set_ylabel('Z (m)')
ax3.set_title('无人机高度变化')
ax3.grid(True, alpha=0.3)
ax3.legend()
plt.tight_layout()
plt.show()