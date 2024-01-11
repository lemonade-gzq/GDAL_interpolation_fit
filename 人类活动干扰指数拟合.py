import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
#
# 数据
data = pd.read_csv(r'E:\城市与区域生态\大熊猫和竹\平武种群动态模拟\三调shap_data.csv', encoding='utf-8')
data = data.iloc[:, :]
data = data.values

variant = data[:, 5:]
shap_value = data[:, 0:5]

variant = variant[:, 2].reshape(-1)
shap_value = shap_value[:, 2].reshape(-1)



variant = np.where(variant == 0, 0.0001, variant)  # 对数函数的定义域为大于0
x_train, x_test, y_train, y_test = train_test_split(variant, shap_value, test_size=0.2, random_state=123)
show_y_max = max(y_test)
show_y_min = min(y_test)


def exponential_func(x, a, b):
    return a * np.exp(b * x)

# 使用指数函数模型进行拟合
popt, pcov = curve_fit(exponential_func, x_train, y_train)
a, b = popt

# 打印拟合指数方程
equation = f'{a:.3f} * exp({b:.3f} * x)'
print('Exponential Regression:')
print(f'Equation: {equation}')
# 计算拟合精度和误差指标
y_test_pred = exponential_func(x_test, a, b)
test_score = r2_score(y_test, y_test_pred)
test_rmse_score = np.sqrt(mean_squared_error(y_test, y_test_pred))
print('exponential Regression:')
print(f'Testing R2 score: {test_score:.3f}')
print(f'Testing RMSE: {test_rmse_score:.3f}\n')

# 生成拟合曲线
x_fit = np.linspace(x_train.min(), x_train.max(), num=1000)
y_fit = exponential_func(x_fit, a, b)

# 绘制拟合曲线和数据点
# 计算要显示的数据点数量
num_points = int(len(x_test) * 0.01)
# 随机选择要显示的数据点的索引
random_indices = np.random.choice(len(x_test), size=num_points, replace=False)
# 根据索引选择要显示的数据点
x_display = x_test[random_indices]
y_display = y_test[random_indices]
plt.scatter(x_display, y_display, color='blue', label='Testing Data', alpha=0.5)
plt.plot(x_fit, y_fit, color='green',)
plt.xlabel('x')
plt.ylabel('y')
plt.ylim(show_y_min, show_y_max)  # 设置y轴显示范围
plt.legend(loc='lower right')
plt.show()

