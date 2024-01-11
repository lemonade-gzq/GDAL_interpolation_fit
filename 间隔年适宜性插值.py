import numpy as np
from osgeo import gdal
from osgeo import osr

# 定义文件路径
# caifa_3 = r"E:\城市与区域生态\大熊猫和竹\平武种群动态模拟\干扰点密度\三调采伐90mGEODESIC4k.tif"
# fangmu_3 = r"E:\城市与区域生态\大熊猫和竹\平武种群动态模拟\干扰点密度\三调放牧90mGEODESIC4k.tif"
# kaikang_3 = r"E:\城市与区域生态\大熊猫和竹\平武种群动态模拟\干扰点密度\三调开矿90mGEODESIC4k.tif"
# toulie_3 = r"E:\城市与区域生态\大熊猫和竹\平武种群动态模拟\干扰点密度\三调偷猎90mGEODESIC4k.tif"
# caifa_4 = r"E:\城市与区域生态\大熊猫和竹\平武种群动态模拟\干扰点密度\四调采伐90mGEODESIC4k.tif"
# fangmu_4 = r"E:\城市与区域生态\大熊猫和竹\平武种群动态模拟\干扰点密度\四调放牧90mGEODESIC4k.tif"
# kaikang_4 = r"E:\城市与区域生态\大熊猫和竹\平武种群动态模拟\干扰点密度\四调开矿90mGEODESIC4k.tif"
# toulie_4 = r"E:\城市与区域生态\大熊猫和竹\平武种群动态模拟\干扰点密度\四调偷猎90mGEODESIC4k.tif"
DNR_4 = r"E:/城市与区域生态/大熊猫和竹/种群动态模拟/dis2015.tif"
DNR_3 = r"E:/城市与区域生态/大熊猫和竹/种群动态模拟/dis2000.tif"


# 读取栅格数据
# ds_caifa_3 = gdal.Open(caifa_3)
# ds_fangmu_3 = gdal.Open(fangmu_3)
# ds_kaikuang_3 = gdal.Open(kaikang_3)
# ds_toulie_3 = gdal.Open(toulie_3)
# ds_caifa_4 = gdal.Open(toulie_4)
# ds_fangmu_4 = gdal.Open(caifa_4)
# ds_kaikuang_4 = gdal.Open(fangmu_4)
# ds_toulie_4 = gdal.Open(kaikang_4)
ds_DNR_4 = gdal.Open(DNR_4)
ds_DNR_3 = gdal.Open(DNR_3)

# 获取栅格数据的信息
cols = ds_DNR_4.RasterXSize
rows = ds_DNR_4.RasterYSize
projection = ds_DNR_4.GetProjection()
geotransform = ds_DNR_4.GetGeoTransform()

# 读取栅格数据的像素值
# data_caifa_3 = ds_caifa_3.GetRasterBand(1).ReadAsArray()
# data_fangmu_3 = ds_fangmu_3.GetRasterBand(1).ReadAsArray()
# data_kaikuang_3 = ds_kaikuang_3.GetRasterBand(1).ReadAsArray()
# data_toulie_3 = ds_toulie_3.GetRasterBand(1).ReadAsArray()
# data_caifa_4 = ds_caifa_4.GetRasterBand(1).ReadAsArray()
# data_fangmu_4 = ds_fangmu_4.GetRasterBand(1).ReadAsArray()
# data_kaikuang_4 = ds_kaikuang_4.GetRasterBand(1).ReadAsArray()
# data_toulie_4 = ds_toulie_4.GetRasterBand(1).ReadAsArray()
data_DNR_4 = ds_DNR_4.GetRasterBand(1).ReadAsArray()
data_DNR_3 = ds_DNR_3.GetRasterBand(1).ReadAsArray()
max_DNR = np.max(data_DNR_4)
data_DNR_4 = data_DNR_4 / max_DNR
data_DNR_3 = data_DNR_3 / max_DNR
data_DNR_3 = np.where(data_DNR_3 < 0, 1, data_DNR_3)
data_DNR_4 = np.where(data_DNR_4 < 0, 1, data_DNR_4)

# 计算时间距离权重
years = 2012 - 1998 - 1  # 计算年份数
weights = np.linspace(0, 1, years)  # 根据年份计算线性权重，可根据需要自定义权重计算方法

# 插值处理
# interpolated_caifa = np.zeros((years, rows, cols))  # 存储插值结果
# interpolated_fangmu = np.zeros((years, rows, cols))  # 存储插值结果
# interpolated_kaikuang = np.zeros((years, rows, cols))  # 存储插值结果
# interpolated_toulie = np.zeros((years, rows, cols))  # 存储插值结果

# for i, year in enumerate(range(1999, 2012)):
#     print(i, year)
#     # 计算权重
#     weight_1998 = 1 - weights[i]
#     weight_2012 = weights[i]
    # 根据权重进行插值计算
#     interpolated_caifa[i] = weight_1998 * data_caifa_3 + weight_2012 * data_caifa_4
#     interpolated_fangmu[i] = weight_1998 * data_fangmu_3 + weight_2012 * data_fangmu_4
#     interpolated_kaikuang[i] = weight_1998 * data_kaikuang_3 + weight_2012 * data_kaikuang_4
#     interpolated_toulie[i] = weight_1998 * data_toulie_3 + weight_2012 * data_toulie_4
# del data_caifa_3, data_caifa_4, data_fangmu_3, data_fangmu_4, data_kaikuang_3, data_kaikuang_4, data_toulie_3, \
#     data_toulie_4
# interpolated_caifa = np.where(interpolated_caifa == 0, 0.0001, interpolated_caifa)  # 对数函数的定义域为大于0
# interpolated_fangmu = np.where(interpolated_fangmu == 0, 0.0001, interpolated_fangmu)  # 对数函数的定义域为大于0
# interpolated_kaikuang = np.where(interpolated_kaikuang == 0, 0.0001, interpolated_kaikuang)  # 对数函数的定义域为大于0
# interpolated_toulie = np.where(interpolated_toulie == 0, 0.0001, interpolated_toulie)  # 对数函数的定义域为大于0

# shap_value_caifa = np.zeros((years, rows, cols))  # 存储插值结果
# shap_value_fangmu = np.zeros((years, rows, cols))  # 存储插值结果
# shap_value_kaikuang = np.zeros((years, rows, cols))  # 存储插值结果
# shap_value_toulie = np.zeros((years, rows, cols))  # 存储插值结果
shap_value_DNR = np.zeros((years, rows, cols))  # 存储插值结果

# for i, year in enumerate(range(1999, 2012)):
#     print(i, year)
#
#     shap_value_caifa[i] = weight_2012 * (0.002 * np.log(interpolated_caifa[i]) + 0.020)
#     shap_value_fangmu[i] = weight_2012 * (0.003 * np.log(interpolated_fangmu[i]) + 0.020)
#     shap_value_kaikuang[i] = weight_1998 * (0.068 * np.log(interpolated_fangmu[i]) + 0.429) + \
#                              weight_2012 * (0.002 * np.log(interpolated_kaikuang[i]) + 0.014)
#     shap_value_toulie[i] = weight_2012 * (0.008 * np.log(interpolated_toulie[i]) + 0.067)
# del interpolated_caifa, interpolated_fangmu, interpolated_kaikuang, interpolated_toulie

for i, year in enumerate(range(1999, 2007)):
    weight_1998 = 1 - weights[i]
    weight_2012 = weights[i]
    shap_value_DNR[i] = weight_1998 * (-0.020 * data_DNR_3 ** 0 + -0.337 * data_DNR_3 ** 1 + 3.828 * data_DNR_3 ** 2 +
                                       -10.881 * data_DNR_3 ** 3 + 12.165 * data_DNR_3 ** 4 + -4.746 * data_DNR_3 ** 5) + \
                        weight_2012 * (-0.015 * data_DNR_4 ** 0 + 0.201 * data_DNR_4 ** 1 - 0.753 * data_DNR_4 ** 2 +
                                       +1.137 * data_DNR_4 ** 3 + -0.622 * data_DNR_4 ** 4 + 0.054 * data_DNR_4 ** 5)

for i, year in enumerate(range(2007, 2012)):
    weight_1998 = 1 - weights[i]
    weight_2012 = weights[i]
    shap_value_DNR[i] =  weight_1998 * (-0.020 * data_DNR_4 ** 0 + -0.337 * data_DNR_4 ** 1 + 3.828 * data_DNR_4 ** 2 +
                                       -10.881 * data_DNR_4 ** 3 + 12.165 * data_DNR_4 ** 4 + -4.746 * data_DNR_4 ** 5) + \
                        weight_2012 * (-0.015 * data_DNR_4 ** 0 + 0.201 * data_DNR_4 ** 1 - 0.753 * data_DNR_4 ** 2 +
                                       +1.137 * data_DNR_4 ** 3 + -0.622 * data_DNR_4 ** 4 + 0.054 * data_DNR_4 ** 5)

rds = gdal.Open("E:/城市与区域生态/大熊猫和竹/平武种群动态模拟/only_ndr/三调无道路概率.tif")
band = rds.GetRasterBand(1)
data_3_primitive = band.ReadAsArray(0, 0, cols, rows)
rds = gdal.Open("E:/城市与区域生态/大熊猫和竹/平武种群动态模拟/only_ndr/四调无道路概率.tif")
band = rds.GetRasterBand(1)
data_4_primitive = band.ReadAsArray(0, 0, cols, rows)

for i, year in enumerate(range(1999, 2012)):
    weight_1998 = 1 - weights[i]
    weight_2012 = weights[i]
    data_medium = np.zeros((rows, cols))
    data_medium = (weight_1998 * data_3_primitive + weight_2012 * data_4_primitive)
    max_value = np.max(data_medium)
    min_value = np.min(data_medium)
    normalized_matrix = (data_medium - min_value) / (max_value - min_value)
    output_file = "E:/城市与区域生态/大熊猫和竹/平武种群动态模拟/only_ndr/中间年份概率/{}概率.tif".format(year)
    driver = gdal.GetDriverByName("GTiff")
    output_ds = driver.Create(output_file, cols, rows, 1, gdal.GDT_Float64)
    output_ds.SetGeoTransform(list(geotransform))
    srs = osr.SpatialReference()
    srs.ImportFromWkt(projection)
    output_ds.SetProjection(srs.ExportToWkt())
    output_ds.GetRasterBand(1).WriteArray(normalized_matrix)
