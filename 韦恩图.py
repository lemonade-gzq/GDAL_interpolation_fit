import matplotlib.pyplot as plt
from matplotlib_venn import venn2
font = {'family': 'simsun', 'size': 16}
plt.rc('font', **font)
# 创建图形对象并设置底色为透明
fig = plt.figure(facecolor='none')
# 设置图形的大小和位置

# 数据集大小
n_A = 1251
n_B = 1009
n_common = 757

# 设置颜色
color_A = '#1b95fc'  # 浅紫色
color_B = '#f66337'  # 浅粉色

# 绘制韦恩图
venn2(subsets=(n_A - n_common, n_B - n_common, n_common), set_labels=('A', 'B'), set_colors=(color_A, color_B))
# 创建图例
labels = ['高等级道路影响范围以外', '县道影响范围以外']
plt.legend(labels=labels, loc='lower right')
# 显示图表
plt.show()