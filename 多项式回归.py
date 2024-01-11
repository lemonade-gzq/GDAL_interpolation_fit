import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
# 数据
data = pd.read_csv(filepath_or_buffer=r'E:\城市与区域生态\大熊猫和竹\道路对大熊猫栖息地的影响\道路距离分析\分段统计表\邛崃适宜点数01.csv')
data = data.iloc[:44, :]
data = data.values

count = data[:, -2]
distance = data[:, -1]

# 将数据集分成训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(distance, count, test_size=0.2, random_state=123)

# 尝试拟合 3 次、4 次、5 次多项式回归模型，并计算拟合精度和误差指标
degrees = [6, 7, 8, 9]
colors = ['green', 'green', 'green', 'green']
fig, axs = plt.subplots(1, 4, figsize=(15, 5))
for i, degree in enumerate(degrees):
    # 使用多项式回归模型进行拟合
    coefficients = np.polyfit(distance, count, deg=degree)

    # 计算拟合精度和误差指标
    y_pred = np.polyval(coefficients, distance)
    # y_test_pred = np.polyval(coefficients, x_test)
    score = r2_score(count, y_pred)
    # test_score = r2_score(y_test, y_test_pred)
    rmse_score = np.sqrt(mean_squared_error(count, y_pred))
    # test_rmse_score = np.sqrt(mean_squared_error(y_test, y_test_pred))
    print(f'{degree}-degree Polynomial Regression:')
    print(f'Training R2 score: {score:.3f}')
    # print(f'Testing R2 score: {test_score:.3f}')
    print(f'Training RMSE: {rmse_score:.3f}')
    # print(f'Testing RMSE: {test_rmse_score:.3f}\n')

    # 生成拟合曲线
    x_fit = np.linspace(x_train.min(), x_train.max(), num=1000)
    y_fit = np.polyval(coefficients, x_fit)

    # 计算拐点
    derivative = np.polyder(coefficients, m=1)  # 一阶导数
    roots = np.roots(derivative)  # 求解一阶导数为0的根
    real_roots = [root.real for root in roots if
                  np.isclose(root.imag, 0) and root.real >= x_train.min() and root.real <= x_train.max()]
    print(f'拐点: ', real_roots)


    # 绘制拟合曲线和原始数据散点图
    axs[i].scatter(distance, count, color='blue')
    # axs[i].scatter(x_test, y_test, color='red', label='Testing Data')
    axs[i].plot(x_fit, y_fit, color=colors[i], label='Polynomial Regression')

    # 添加图例和标签
    axs[i].set_xlabel('Distance')
    axs[i].set_ylabel('Count')
    axs[i].set_title(f'{degree}-degree Polynomial Regression')
    axs[i].legend()

    # 添加拐点和拟合方程

    for root in real_roots:
        axs[i].axvline(root, color='black', linestyle='--')
        # axs[i].text(root, y_train.max() * 0.95, f'x={root:.2f}', fontsize=14)


# 显示图形和拟合指标
plt.show()
