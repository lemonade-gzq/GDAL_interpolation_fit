import os

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


filePath = r'E:\城市与区域生态\大熊猫和竹\道路对大熊猫栖息地的影响\道路距离分析\适宜性分析数据准备\xgb和maxent数据准备'

filename = os.listdir(filePath)

data_all = np.zeros((1080 * 924, 24))
m = 24 - 1
for i in filename:
    # if i.split(".")[-1] == "tif" and i != "2010ES.tif" and i != "2000dis.tif" and i != "2015dis.tif" and i != "2015firstdis.tif" and i != "2015seconddis.tif":
    if i.split(".")[-1] == "tif" and i != "2000ES.tif" and i != "2000dis.tif" and i != "2015firstdis.tif" and i != "2000firstdis.tif" and i != "2000seconddis.tif" and i != "2015seconddis.tif":
        rds = gdal.Open(filePath + '\\' + i)
        cols = rds.RasterXSize
        rows = rds.RasterYSize
        geotransform = rds.GetGeoTransform()  # geotransform
        projection = rds.GetProjectionRef()  # projection
        band = rds.GetRasterBand(1)
        print(cols, rows)
        print(rds.GetGeoTransform())
        var_name = i.split('_')[-1].split('.')[0]
        print(var_name)
        globals()[var_name] = band.ReadAsArray(0, 0, cols, rows)
        for i in range(rows):
            for j in range(cols):
                data_all[i * cols + j, m] = globals()[var_name][i, j]
        m -= 1

np.nan_to_num(data_all, nan=-9999, copy=False)

data = pd.read_csv(r"E:\城市与区域生态\大熊猫和竹\道路对大熊猫栖息地的影响\道路距离分析\适宜性分析数据准备\xgb和maxent数据准备\13.csv",
                   engine='python', header=0)
data = data.iloc[:, :]
data.info()
X, Y = data[[x for x in data.columns if x != 'label' and x != 'FID'and x != '2015seconddis'and x != '2015firstdis']], data['label']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=5)
predict_X = data_all[:, :]

# 使用RandomForestRegressor训练模型，并对测试数据做出预测，结果存储在变量y_pred中RandomForestClassifier

clf = XGBClassifier(objective="binary:logistic", seed=1024, )
"""learning_rate=0.1, max_depth=13, min_child_weight=1,
                    gamma=0, subsample=0.78, colsample_bytree=0.57, reg_alpha=1e-5, reg_lambda=0.09, n_estimators=44"""
clf.fit(X_train, y_train)
train_predict = clf.predict(X_train)
test_predict = clf.predict(X_test)

cols_feature = ['slope', 'dem', 'bio9', 'bio8', 'bio7', 'bio6', 'bio5', 'bio4', 'bio3', 'bio2', 'bio19',
                'bio18', 'bio17', 'bio16', 'bio15', 'bio14', 'bio13', 'bio12', 'bio11', 'bio10', 'bio1', 'aspect',
                '2015dis', '2010ES' ]  #
explainer = shap.TreeExplainer(clf, X_train, feature_perturbation="interventional", model_output='probability')
shap_values = explainer.shap_values(X_train[cols_feature])

shap.summary_plot(shap_values, X_train[cols_feature])
shap.summary_plot(shap_values, X_train[cols_feature], plot_type="bar")

shap.dependence_plot('2015firstdis', shap_values, X_train, interaction_index=None, show=False)
shap.dependence_plot('2015seconddis', shap_values, X_train, interaction_index=None, show=False)

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
print(data_probability)
data_probability.shape = (rows, cols)
driver = gdal.GetDriverByName('GTiff')
# dst_filename = r"E:\城市与区域生态\大熊猫和竹\种群动态模拟_lstm\四调概率输出10xgb_lstm.tif"
dst_ds = driver.Create(r"E:\城市与区域生态\大熊猫和竹\道路对大熊猫栖息地的影响\道路距离分析\适宜性分析数据准备\全山系适宜性\四调概率输出dis.tif", cols, rows, 1,
                       gdal.GDT_Float64)
dst_ds.SetGeoTransform(list(geotransform))
srs = osr.SpatialReference()
# srs.SetWellKnownGeogCS('EPSG:32648')
srs.ImportFromWkt(projection)
dst_ds.SetProjection(srs.ExportToWkt())
dst_ds.GetRasterBand(1).WriteArray(data_probability)
