"""
结合点位与道路的距离计算每个栅格的适宜性情况，被认为不可比
"""
from osgeo import gdal
from osgeo import osr
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

from matplotlib import pyplot

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.datasets import load_iris
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier
import seaborn as sns
import matplotlib
import sklearn.model_selection as ms
import shap
from sklearn.preprocessing import LabelEncoder

# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

# rc = {'font.sans-serif': ['Times New Roman']}
# sns.set( font_scale=1.5)
plt.figure(dpi=150)


def kappa(confusion_matrix):
    """计算kappa值系数"""
    pe_rows = np.sum(confusion_matrix, axis=0)
    pe_cols = np.sum(confusion_matrix, axis=1)
    sum_total = sum(pe_cols)
    pe = np.dot(pe_rows, pe_cols) / float(sum_total ** 2)
    po = np.trace(confusion_matrix) / float(sum_total)
    return (po - pe) / (1 - pe)


def f1(confusion_matrix):
    p = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[0][1])
    r = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[1][0])
    return 2 * (p * r) / (p + r)


rds = gdal.Open(r"E:\城市与区域生态\气候数据\六山系矩形\dem.tif")
cols = rds.RasterXSize
rows = rds.RasterYSize
band = rds.GetRasterBand(1)
print(cols, rows)  # 414 266
print(rds.GetGeoTransform())  # (103.223472222222, 0.0002777777779999902, 0.0, 31.147361111111, 0.0,
# -0.0002777777779999927)
data_dem = band.ReadAsArray(0, 0, cols, rows)

rds = gdal.Open(r"E:\城市与区域生态\气候数据\六山系矩形\slope.tif")
cols = rds.RasterXSize
rows = rds.RasterYSize
band = rds.GetRasterBand(1)
print(cols, rows)  # 414 266
print(rds.GetGeoTransform())  # (103.223472222222, 0.0002777777779999902, 0.0, 31.147361111111, 0.0,
# -0.0002777777779999927)
data_slope = band.ReadAsArray(0, 0, cols, rows)

rds = gdal.Open(r"E:\城市与区域生态\气候数据\六山系矩形\aspect.tif")
cols = rds.RasterXSize
rows = rds.RasterYSize
band = rds.GetRasterBand(1)
print(cols, rows)  # 414 266
print(rds.GetGeoTransform())  # (103.223472222222, 0.0002777777779999902, 0.0, 31.147361111111, 0.0,
# -0.0002777777779999927)
data_aspect = band.ReadAsArray(0, 0, cols, rows)
# print(data_dem)

rds = gdal.Open(r"E:\城市与区域生态\其他\20230813阻力面\biosys10.tif")  # 00 15
cols = rds.RasterXSize
rows = rds.RasterYSize
band = rds.GetRasterBand(1)
print(cols, rows)  # 414 266
print(rds.GetGeoTransform())  # (103.223472222222, 0.0002777777779999902, 0.0, 31.147361111111, 0.0,
# -0.0002777777779999927)
data_biosys = band.ReadAsArray(0, 0, cols, rows)

rds = gdal.Open(r"E:\城市与区域生态\气候数据\六山系矩形\bio1.tif")
cols = rds.RasterXSize
rows = rds.RasterYSize
band = rds.GetRasterBand(1)
print(cols, rows)  # 414 266
print(rds.GetGeoTransform())  # (103.223472222222, 0.0002777777779999902, 0.0, 31.147361111111, 0.0,
# -0.0002777777779999927)
data_bio1 = band.ReadAsArray(0, 0, cols, rows)

rds = gdal.Open(r"E:\城市与区域生态\气候数据\六山系矩形\bio2.tif")
cols = rds.RasterXSize
rows = rds.RasterYSize
band = rds.GetRasterBand(1)
print(cols, rows)  # 414 266
print(rds.GetGeoTransform())  # (103.223472222222, 0.0002777777779999902, 0.0, 31.147361111111, 0.0,
# -0.0002777777779999927)
data_bio2 = band.ReadAsArray(0, 0, cols, rows)

rds = gdal.Open(r"E:\城市与区域生态\气候数据\六山系矩形\bio3.tif")
cols = rds.RasterXSize
rows = rds.RasterYSize
band = rds.GetRasterBand(1)
print(cols, rows)  # 414 266
print(rds.GetGeoTransform())  # (103.223472222222, 0.0002777777779999902, 0.0, 31.147361111111, 0.0,
# -0.0002777777779999927)
data_bio3 = band.ReadAsArray(0, 0, cols, rows)

rds = gdal.Open(r"E:\城市与区域生态\气候数据\六山系矩形\bio4.tif")
cols = rds.RasterXSize
rows = rds.RasterYSize
band = rds.GetRasterBand(1)
print(cols, rows)  # 414 266
print(rds.GetGeoTransform())  # (103.223472222222, 0.0002777777779999902, 0.0, 31.147361111111, 0.0,
# -0.0002777777779999927)
data_bio4 = band.ReadAsArray(0, 0, cols, rows)

rds = gdal.Open(r"E:\城市与区域生态\气候数据\六山系矩形\bio5.tif")
cols = rds.RasterXSize
rows = rds.RasterYSize
band = rds.GetRasterBand(1)
print(cols, rows)  # 414 266
print(rds.GetGeoTransform())  # (103.223472222222, 0.0002777777779999902, 0.0, 31.147361111111, 0.0,
# -0.0002777777779999927)
data_bio5 = band.ReadAsArray(0, 0, cols, rows)

rds = gdal.Open(r"E:\城市与区域生态\气候数据\六山系矩形\bio6.tif")
cols = rds.RasterXSize
rows = rds.RasterYSize
band = rds.GetRasterBand(1)
print(cols, rows)  # 414 266
print(rds.GetGeoTransform())  # (103.223472222222, 0.0002777777779999902, 0.0, 31.147361111111, 0.0,
# -0.0002777777779999927)
data_bio6 = band.ReadAsArray(0, 0, cols, rows)

rds = gdal.Open(r"E:\城市与区域生态\气候数据\六山系矩形\bio7.tif")
cols = rds.RasterXSize
rows = rds.RasterYSize
band = rds.GetRasterBand(1)
print(cols, rows)  # 414 266
print(rds.GetGeoTransform())  # (103.223472222222, 0.0002777777779999902, 0.0, 31.147361111111, 0.0,
# -0.0002777777779999927)
data_bio7 = band.ReadAsArray(0, 0, cols, rows)

rds = gdal.Open(r"E:\城市与区域生态\气候数据\六山系矩形\bio8.tif")
cols = rds.RasterXSize
rows = rds.RasterYSize
band = rds.GetRasterBand(1)
print(cols, rows)  # 414 266
print(rds.GetGeoTransform())  # (103.223472222222, 0.0002777777779999902, 0.0, 31.147361111111, 0.0,
# -0.0002777777779999927)
data_bio8 = band.ReadAsArray(0, 0, cols, rows)

rds = gdal.Open(r"E:\城市与区域生态\气候数据\六山系矩形\bio9.tif")
cols = rds.RasterXSize
rows = rds.RasterYSize
band = rds.GetRasterBand(1)
print(cols, rows)  # 414 266
print(rds.GetGeoTransform())  # (103.223472222222, 0.0002777777779999902, 0.0, 31.147361111111, 0.0,
# -0.0002777777779999927)
data_bio9 = band.ReadAsArray(0, 0, cols, rows)

rds = gdal.Open(r"E:\城市与区域生态\气候数据\六山系矩形\bio10.tif")
cols = rds.RasterXSize
rows = rds.RasterYSize
band = rds.GetRasterBand(1)
print(cols, rows)  # 414 266
print(rds.GetGeoTransform())  # (103.223472222222, 0.0002777777779999902, 0.0, 31.147361111111, 0.0,
# -0.0002777777779999927)
data_bio10 = band.ReadAsArray(0, 0, cols, rows)

rds = gdal.Open(r"E:\城市与区域生态\气候数据\六山系矩形\bio11.tif")
cols = rds.RasterXSize
rows = rds.RasterYSize
band = rds.GetRasterBand(1)
print(cols, rows)  # 414 266
print(rds.GetGeoTransform())  # (103.223472222222, 0.0002777777779999902, 0.0, 31.147361111111, 0.0,
# -0.0002777777779999927)
data_bio11 = band.ReadAsArray(0, 0, cols, rows)

rds = gdal.Open(r"E:\城市与区域生态\气候数据\六山系矩形\bio12.tif")
cols = rds.RasterXSize
rows = rds.RasterYSize
band = rds.GetRasterBand(1)
print(cols, rows)  # 414 266
print(rds.GetGeoTransform())  # (103.223472222222, 0.0002777777779999902, 0.0, 31.147361111111, 0.0,
# -0.0002777777779999927)
data_bio12 = band.ReadAsArray(0, 0, cols, rows)

rds = gdal.Open(r"E:\城市与区域生态\气候数据\六山系矩形\bio13.tif")
cols = rds.RasterXSize
rows = rds.RasterYSize
band = rds.GetRasterBand(1)
print(cols, rows)  # 414 266
print(rds.GetGeoTransform())  # (103.223472222222, 0.0002777777779999902, 0.0, 31.147361111111, 0.0,
# -0.0002777777779999927)
data_bio13 = band.ReadAsArray(0, 0, cols, rows)

rds = gdal.Open(r"E:\城市与区域生态\气候数据\六山系矩形\bio14.tif")
cols = rds.RasterXSize
rows = rds.RasterYSize
band = rds.GetRasterBand(1)
print(cols, rows)  # 414 266
print(rds.GetGeoTransform())  # (103.223472222222, 0.0002777777779999902, 0.0, 31.147361111111, 0.0,
# -0.0002777777779999927)
data_bio14 = band.ReadAsArray(0, 0, cols, rows)

rds = gdal.Open(r"E:\城市与区域生态\气候数据\六山系矩形\bio15.tif")
cols = rds.RasterXSize
rows = rds.RasterYSize
band = rds.GetRasterBand(1)
print(cols, rows)  # 414 266
print(rds.GetGeoTransform())  # (103.223472222222, 0.0002777777779999902, 0.0, 31.147361111111, 0.0,
# -0.0002777777779999927)
data_bio15 = band.ReadAsArray(0, 0, cols, rows)

rds = gdal.Open(r"E:\城市与区域生态\气候数据\六山系矩形\bio16.tif")
cols = rds.RasterXSize
rows = rds.RasterYSize
band = rds.GetRasterBand(1)
print(cols, rows)  # 414 266
print(rds.GetGeoTransform())  # (103.223472222222, 0.0002777777779999902, 0.0, 31.147361111111, 0.0,
# -0.0002777777779999927)
data_bio16 = band.ReadAsArray(0, 0, cols, rows)

rds = gdal.Open(r"E:\城市与区域生态\气候数据\六山系矩形\bio17.tif")
cols = rds.RasterXSize
rows = rds.RasterYSize
band = rds.GetRasterBand(1)
print(cols, rows)  # 414 266
print(rds.GetGeoTransform())  # (103.223472222222, 0.0002777777779999902, 0.0, 31.147361111111, 0.0,
# -0.0002777777779999927)
data_bio17 = band.ReadAsArray(0, 0, cols, rows)

rds = gdal.Open(r"E:\城市与区域生态\气候数据\六山系矩形\bio18.tif")
cols = rds.RasterXSize
rows = rds.RasterYSize
band = rds.GetRasterBand(1)
print(cols, rows)  # 414 266
print(rds.GetGeoTransform())  # (103.223472222222, 0.0002777777779999902, 0.0, 31.147361111111, 0.0,
# -0.0002777777779999927)
data_bio18 = band.ReadAsArray(0, 0, cols, rows)

rds = gdal.Open(r"E:\城市与区域生态\气候数据\六山系矩形\bio19.tif")
cols = rds.RasterXSize
rows = rds.RasterYSize
band = rds.GetRasterBand(1)
print(cols, rows)  # 414 266
print(rds.GetGeoTransform())  # (103.223472222222, 0.0002777777779999902, 0.0, 31.147361111111, 0.0,
# -0.0002777777779999927)
data_bio19 = band.ReadAsArray(0, 0, cols, rows)

rds = gdal.Open(r"E:\城市与区域生态\其他\20230813阻力面\dis2road.tif")  # dis2000 dis2015
geotransform = rds.GetGeoTransform()  # geotransform
projection = rds.GetProjectionRef()  # projection
cols = rds.RasterXSize
rows = rds.RasterYSize
band = rds.GetRasterBand(1)
print(cols, rows)  # 414 266
print(rds.GetGeoTransform())  # (103.223472222222, 0.0002777777779999902, 0.0, 31.147361111111, 0.0,
# -0.0002777777779999927)
data_dis = band.ReadAsArray(0, 0, cols, rows)

data_all = np.zeros((cols * rows, 24))
for i in range(rows):
    for j in range(cols):
        data_all[i * cols + j, 0] = data_slope[i, j]
        data_all[i * cols + j, 1] = data_dem[i, j]
        data_all[i * cols + j, 2] = data_biosys[i, j]
        data_all[i * cols + j, 3] = data_bio9[i, j]
        data_all[i * cols + j, 4] = data_bio8[i, j]
        data_all[i * cols + j, 5] = data_bio7[i, j]
        data_all[i * cols + j, 6] = data_bio6[i, j]
        data_all[i * cols + j, 7] = data_bio5[i, j]
        data_all[i * cols + j, 8] = data_bio4[i, j]
        data_all[i * cols + j, 9] = data_bio3[i, j]
        data_all[i * cols + j, 10] = data_bio2[i, j]
        data_all[i * cols + j, 11] = data_bio19[i, j]
        data_all[i * cols + j, 12] = data_bio18[i, j]
        data_all[i * cols + j, 13] = data_bio17[i, j]
        data_all[i * cols + j, 14] = data_bio16[i, j]
        data_all[i * cols + j, 15] = data_bio15[i, j]
        data_all[i * cols + j, 16] = data_bio14[i, j]
        data_all[i * cols + j, 17] = data_bio13[i, j]
        data_all[i * cols + j, 18] = data_bio12[i, j]
        data_all[i * cols + j, 19] = data_bio11[i, j]
        data_all[i * cols + j, 20] = data_bio10[i, j]
        data_all[i * cols + j, 21] = data_bio1[i, j]
        data_all[i * cols + j, 22] = data_aspect[i, j]
        data_all[i * cols + j, 23] = data_dis[i, j]

# print(data_all[:, 2])
data = pd.read_csv(r"E:\城市与区域生态\其他\20230813阻力面\样本点.csv", engine='python', header=0)
data = data.iloc[:, 1:]
# data = data[['slope', 'dem', 'aspect', 'bio14', 'ecosystem', 'dis2000', 'label']]
data.info()

X, Y = data[[x for x in data.columns if x != 'label' and x != 'id']], data['label']
print(X, Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=5, shuffle=True)

params = {'objective': 'reg:logistic', 'booster': 'gbtree', 'silent': 1}
# seed=1024, learning_rate=0.2, n_estimators=25, max_depth=4, min_child_weight=2, gamma=1.0,
#                         colsample_bytree=1.0, subsample=1.0, reg_alpha=0
clf = XGBClassifier(objective="binary:logistic", seed=1024, learning_rate=0.1, max_depth=4, min_child_weight=2,
                    gamma=0.98, subsample=1.0, colsample_bytree=0.89, alpha=0.03, reg_lambda=0.53, n_estimators=95)

clf.fit(X_train, y_train)
train_predict = clf.predict(X_train)
test_predict = clf.predict(X_test)

# 利用accuracy（准确度）【预测正确的样本数目占总预测样本数目的比例】评估模型效果
print('The accuracy of the Logistic Regression is:', metrics.accuracy_score(y_test, test_predict))
print("AUC 得分 (测试集): %f" % metrics.roc_auc_score(y_test, test_predict))
print("MSE:", mean_squared_error(y_test, test_predict))
print("MAE:", mean_absolute_error(y_test, test_predict))
print("R2:", r2_score(y_test, test_predict))
print("acc:", accuracy_score(y_test, test_predict))
print(clf.feature_importances_)
pyplot.bar(range(len(clf.feature_importances_)), clf.feature_importances_)
pyplot.show()

# 查看混淆矩阵 (预测值和真实值的各类情况统计矩阵)
confusion_matrix_result = metrics.confusion_matrix(test_predict, y_test)
print('The confusion matrix result:\n', confusion_matrix_result)
print("kappa for balance matrix: %s" % kappa(confusion_matrix_result))
print("F1", f1(confusion_matrix_result))
# 利用热力图对于结果进行可视化
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_result, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

plot_importance(clf, max_num_features=24, importance_type='gain')
plt.show()
# predict_X = pd.DataFrame(data_all[:, :], columns=['slope', 'dem', 'aspect',  'bio14',  'hh', 'dis2000'])
predict_X = pd.DataFrame(data_all[:, :])

# predict_X = xgb.DMatrix(predict_X)
probability_Y = clf.predict_proba(predict_X)  # 分类概率
print(probability_Y.shape)
f1_score = ms.cross_val_score(clf, X, Y, cv=5, scoring="f1")  # f1得分
print(f"F1分数为：{np.mean(f1_score)}")
f1_score = ms.cross_val_score(clf, X, Y, cv=5, scoring="r2")  # f1得分
print(f"r2分数为：{np.mean(f1_score)}")

data_probability = probability_Y[:, 1]
print(len(data_probability))
for i in range(len(data_probability)):
    data_probability[i] = pow(1000, -data_probability[i])

data_probability.shape = (rows, cols)
driver = gdal.GetDriverByName('GTiff')
# dst_filename = r"E:\城市与区域生态\大熊猫和竹\种群动态模拟_lstm\四调概率输出10xgb_lstm.tif"
dst_ds = driver.Create(r"E:\城市与区域生态\其他\20230813阻力面\阻力.tif", cols, rows, 1, gdal.GDT_Float64)
dst_ds.SetGeoTransform(list(geotransform))
srs = osr.SpatialReference()
# srs.SetWellKnownGeogCS('EPSG:32648')
srs.ImportFromWkt(projection)
dst_ds.SetProjection(srs.ExportToWkt())
dst_ds.GetRasterBand(1).WriteArray(data_probability)
