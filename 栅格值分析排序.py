from osgeo import gdal, osr
import numpy as np
from osgeo import ogr
from collections import Counter
import os


tif_path = []
files = os.listdir("E:\\城市与区域生态\\大熊猫和竹\\道路对大熊猫栖息地的影响\\道路距离分析\\maxent适宜性")
for name in files:
    # if name.endswith('.tif'):
    if name == "13年平均适宜性.tif":
        path = "E:\\城市与区域生态\\大熊猫和竹\\道路对大熊猫栖息地的影响\\道路距离分析\\maxent适宜性\\" + name
        rds = gdal.Open(path)
        cols = rds.RasterXSize
        rows = rds.RasterYSize
        print(cols, rows)
        geotransform = rds.GetGeoTransform()
        band = rds.GetRasterBand(1)
        data = band.ReadAsArray(0, 0)

        data_1d = data.reshape(cols * rows, )  # 一维数组

        data_list = data_1d.tolist()  # 一维列表可哈希
        nodata = data_list[0]
        # print(data)
        value_count = Counter(data_list)  # 键值对为value——count

        del value_count[nodata]  # 删除无效值
        # print(value_count)
        # print("-"*20)

        value_order_count = sorted(zip(value_count.keys(), value_count.values()))  # 对（值键对）排序，并返回一个元素为元组的列表
        print("值键对排序列表", value_order_count)
        # print(value_order_multiply[1])
        # print(type(value_order_multiply))
        print("-" * 20)

        value_multiply = {}  # value * count并放入该字典的值中
        # for key, value in value_order_count.items():
        for i in range(len(value_order_count)):
            value_multiply[value_order_count[i][0]] = value_order_count[i][0] * value_order_count[i][1]
            # value_multiply[key] = key * value
        print("value_multiply,乘积字典", value_multiply)
        print("-" * 20)


        order_multiply = []  # 提取排好序的value*count
        for value in value_multiply.values():
            order_multiply.append(value)
        print("排好序的乘积列表", order_multiply)
        print("-" * 20)
        order_multiply = np.array(order_multiply)
        order_accumulate = np.cumsum(order_multiply)  # 每个元素和前面的所有元素相加
        print("累加之后的数组", order_accumulate)
        print("-" * 20)

        # level_1, level_2, level_3, level_4 = [], [], [], []
        level = dict()
        # value_order_multiply = dict(value_order_multiply)
        value_count = list(value_count.items())
        Threshold50 = order_accumulate[-1] * 0.5
        Threshold80 = order_accumulate[-1] * 0.8
        for i in range(len(order_accumulate)):
            if (order_accumulate[i] < Threshold80) and (order_accumulate[i] >= Threshold50):
                # level_2.append(value_count[i][0])
                level[value_order_count[i][0]] = 2
            elif (order_accumulate[i] < Threshold50) and (order_accumulate[i] >= 0):
                # level_1.append(value_count[i][0])
                level[value_order_count[i][0]] = 1
            elif (order_accumulate[i] < order_accumulate[-1]) and (order_accumulate[i] >= Threshold80):
                # level_4.append(value_count[i][0])
                level[value_order_count[i][0]] = 3
        print("level：", level)
        # min_level, max_level = min(level_1), max(level_1)
        # level_1.clear()
        # level_1 = [min_level, max_level]
        # level_1 = np.asarray(level_1)
        # np.save(path.split(sep=".")[-2] + "_level1.npy", level_1)
        #
        # min_level, max_level = min(level_2), max(level_2)
        # level_2.clear()
        # level_2 = [min_level, max_level]
        # level_2 = np.array(level_2)
        # np.save(path.split(sep=".")[-2] + "_level2.npy", level_2)
        #
        # min_level, max_level = min(level_3), max(level_3)
        # level_3.clear()
        # level_3 = [min_level, max_level]
        # level_3 = np.array(level_3)
        # np.save(path.split(sep=".")[-2] + "_level3.npy", level_3)
        #
        # min_level, max_level = min(level_4), max(level_4)
        # level_4.clear()
        # level_4 = [min_level, max_level]
        # level_4 = np.array(level_4)
        # np.save(path.split(sep=".")[-2] + "_level4.npy", level_4)
        # newdata = np.zeros_like(data)
        # for i in range(data.shape[0]):
        #     for j in range(data.shape[1]):
        #         if data[i][j] in level_1:
        #             newdata[i][j] = 1
        #         elif data[i][j] in level_2:
        #             newdata[i][j] = 2
        #         elif data[i][j] in level_3:
        #             newdata[i][j] = 3
        #         else:
        #             if data[i][j] != nodata:
        #                 newdata[i][j] = 4
        newdata = np.vectorize(level.get)(data)  # 将一个只能对一个值进行操作的函数扩展到整个数组上
        driver = gdal.GetDriverByName('GTiff')
        out = "E:\\城市与区域生态\\大熊猫和竹\\道路对大熊猫栖息地的影响\\道路距离分析\\maxent适宜性\\rc\\" + name.split(".")[0] + "rc.tif"
        dst_ds = driver.Create(out, cols, rows, 1, gdal.GDT_Int32)
        dst_ds.SetGeoTransform(list(geotransform))
        srs = osr.SpatialReference()
        srs.SetWellKnownGeogCS('EPSG:4326')
        dst_ds.SetProjection(srs.ExportToWkt())
        dst_ds.GetRasterBand(1).WriteArray(newdata)
