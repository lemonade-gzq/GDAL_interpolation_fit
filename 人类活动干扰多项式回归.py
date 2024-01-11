import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# 设置全局字体样式
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14
# 数据
data = pd.read_csv(r'E:\城市与区域生态\大熊猫和竹\平武种群动态模拟\三调DNRshap_data.csv', encoding='utf-8')
data = data.iloc[:, :]
data = data.values

variant = data[:, 0]
shap_value = data[:,1]

# variant = variant[:, 0].reshape(-1)
# shap_value = shap_value[:, 0].reshape(-1)
max_variant = max(variant)
print('max:', max_variant)
min_variant = min(variant)
print('min:', min_variant)
# 寻找要删除的行的索引
delete_indices = (variant == min_variant)
# 保留不等于 min_variant 的行
variant = variant[~delete_indices]
shap_value = shap_value[~delete_indices]
Standard_variant = variant / max_variant
# 将数据集分成训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(Standard_variant, shap_value, test_size=0.2, random_state=123)
show_y_max = max(y_test)
show_y_min = min(y_test)

# 尝试拟合 3 次、4 次、5 次多项式回归模型，并计算拟合精度和误差指标
degrees = [5]
colors = ['green']
fig, axs = plt.subplots(1, 1, figsize=(8,6))
for i, degree in enumerate(degrees):
    # 使用多项式回归模型进行拟合
    coefficients = np.polyfit(x_train, y_train, deg=degree)
    # 打印拟合方程
    equation = ' + '.join(f'{coefficients[j]:.3f} * x^{degree - j}' for j in range(degree + 1)[::-1])
    print(f'{degree}-degree Polynomial Regression:')
    print(f'Equation: {equation}')

    # 计算拟合精度和误差指标
    y_test_pred = np.polyval(coefficients, x_test)
    test_score = r2_score(y_test, y_test_pred)
    test_rmse_score = np.sqrt(mean_squared_error(y_test, y_test_pred))
    print(f'{degree}-degree Polynomial Regression:')
    print(f'Testing R2 score: {test_score:.3f}')
    print(f'Testing RMSE: {test_rmse_score:.3f}\n')


    # 生成拟合曲线
    x_fit = np.linspace(x_train.min(), x_train.max(), num=1000)
    y_fit = np.polyval(coefficients, x_fit)
    # 计算要显示的数据点数量
    num_points = int(len(x_test) * 0.01)
    # 随机选择要显示的数据点的索引
    random_indices = np.random.choice(len(x_test), size=num_points, replace=False)

    # 根据索引选择要显示的数据点
    x_display = x_test[random_indices]
    y_display = y_test[random_indices]
    axs.scatter(Standard_variant, shap_value, color='blue', alpha=0.5)
    axs.plot(x_fit, y_fit, color=colors[i])
    # 添加拟合函数和R2值的文本注释
    # annotation = f'-0.013-0.299x+2.832$x^2$-7.156$x^3$+7.375$x^4$-2.711$x^5$\n$R^2$ : {test_score:.3f}'
    #四调-0.015+0.032x+2.160$x^2$-0.148$x^3$-0.114$x^4$+0.104$x^5$
    #三调-0.013-0.299x+2.832$x^2$-7.156$x^3$+7.375$x^4$-2.711$x^5$
    # axs.text(0.03, 0.85, annotation, transform=axs.transAxes, fontsize=20)

    # 添加图例和标签
    axs.set_xlabel('DNR')
    axs.set_ylabel('shap_value')
    axs.set_ylim(show_y_min, show_y_max)  # 设置y轴显示范围
    # axs.set_title(f'{degree}-degree Polynomial Regression')
plt.show()
