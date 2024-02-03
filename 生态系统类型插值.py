#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from osgeo import gdal
import random
from osgeo import osr
#E:/城市与区域生态/大熊猫和竹/种群动态模拟/岷山竹子分布/岷山四调竹子分布.tif
ECO_2015 = r"E:/城市与区域生态/大熊猫和竹/种群动态模拟/eco2015.tif"
ECO_2000 = r"E:/城市与区域生态/大熊猫和竹/种群动态模拟/eco2000.tif"

eco_4 = gdal.Open(ECO_2015)
eco_3 = gdal.Open(ECO_2000)

# 获取栅格数据的信息
cols = eco_4.RasterXSize
rows = eco_4.RasterYSize
projection = eco_4.GetProjection()
geotransform = eco_4.GetGeoTransform()
data_eco_4 = eco_4.GetRasterBand(1).ReadAsArray()
data_eco_3 = eco_3.GetRasterBand(1).ReadAsArray()

data_change = data_eco_4 - data_eco_3
data_change = np.where(data_change != 0, random.randint(1, 13), data_change)

for i in range(1, 14):
    mask = np.where(data_change == i)
    data_eco_3[mask] = data_eco_4[mask]
    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.Create(r"E:\城市与区域生态\大熊猫和竹\平武种群动态模拟\base3onlydnr\eco逐年变化\{}年eco.tif".format(i + 1998), cols, rows, 1,
                           gdal.GDT_UInt16)
    dst_ds.SetGeoTransform(list(geotransform))
    srs = osr.SpatialReference()
    srs.ImportFromWkt(projection)
    dst_ds.SetProjection(srs.ExportToWkt())
    dst_ds.GetRasterBand(1).WriteArray(data_eco_3)
