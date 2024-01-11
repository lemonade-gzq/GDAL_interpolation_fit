from osgeo import gdal
import numpy as np
import pandas as pd
import numba

from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor

# df = np.ones((905 * 741, 24))
# df = pd.DataFrame(df)
# df.columns = ['bio1', 'bio2', 'bio3', 'bio4', 'bio5', 'bio6', 'bio7', 'bio8', 'bio9', 'bio10', 'bio11', 'bio12',
#               'bio13', 'bio14', 'bio15', 'bio16', 'bio17', 'bio18', 'bio19', 'dem', 'slope', 'aspect', 'hh', 'const']
# for i in range(23):
#     name = "bio{}.tif".format(i + 1)
#     path = "F:\\climate\\vif\\" + name
#     rds = gdal.Open(path)
#     cols = rds.RasterXSize
#     rows = rds.RasterYSize
#     print(cols, rows)
#     geotransform = rds.GetGeoTransform()
#     band = rds.GetRasterBand(1)
#     data = band.ReadAsArray(0, 0).astype(float)
#     data = data.reshape(cols * rows, 1)
#     data[np.where(data == data[0])] = np.nan
#     df[df.columns[i]] = data
# df = df.dropna(axis=0, how='any')  # 删除带有任何空值的行
# # 计算方差膨胀因子
# vif_list = []
# x = np.array(df)
# for i in range(x.shape[1]):
#     vif_list.append(variance_inflation_factor(x, i))
# df_vif = pd.DataFrame({'variable': list(df.columns), 'vif': vif_list})
# # df_vif = df_vif[~(df_vif['variable'] == 19)]  # 删除常数项
# print(df_vif)

data = pd.read_csv(filepath_or_buffer=r"E:\城市与区域生态\大熊猫和竹\平武种群动态模拟\方差膨胀因子计算DNR.csv", header=0)
data = data.loc[:, "bio14":"bamboo3"]
print(data)
vif_list = []
name = data.columns
x = np.matrix(data)
vif_list = [variance_inflation_factor(x, i) for i in range(x.shape[1])]
vif = pd.DataFrame({'feature': name, 'VIF': vif_list})
print(vif)

#
# def vif(df,col_i):
#     from statsmodels.formula.api import ols
#
#     cols = list(df.columns)
#     cols_noti = cols
#     formula = col_i + '~' + '+'.join(cols_noti)
#     print(formula)
#     r2 = ols(formula, df).fit().rsquared
#     return 1. / (1. - r2)
#
#
# for i in data.columns:
#     print(i, '\t', vif(df=data, col_i=i))
