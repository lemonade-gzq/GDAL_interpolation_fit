from osgeo import gdal, osr
import numpy as np
from osgeo import ogr
from collections import Counter
import os


path = "E:\\城市与区域生态\\大熊猫和竹\\道路对大熊猫栖息地的影响\\道路距离分析\\核密度\\p01核密度T.tif"
rds = gdal.Open(path)
cols = rds.RasterXSize
rows = rds.RasterYSize
print(cols, rows)
geotransform = rds.GetGeoTransform()
band = rds.GetRasterBand(1)
data = band.ReadAsArray(0, 0)
reclass = {}
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if data[i][j] <= 0.0004:
            reclass[data[i][j]] = 0
        else:
            reclass[data[i][j]] = data[i][j]
# reclass = sorted(zip(reclass.keys(), reclass.values()), reverse=True)
newdata = np.vectorize(reclass.get)(data)  # 将一个只能对一个值进行操作的函数扩展到整个数组上
driver = gdal.GetDriverByName('GTiff')
out = "E:\\城市与区域生态\\大熊猫和竹\\道路对大熊猫栖息地的影响\\道路距离分析\\核密度\\p01核密度R.tif"
dst_ds = driver.Create(out, cols, rows, 1, gdal.GDT_Int32)
dst_ds.SetGeoTransform(list(geotransform))
srs = osr.SpatialReference()
srs.SetWellKnownGeogCS('EPSG:4326')
dst_ds.SetProjection(srs.ExportToWkt())
dst_ds.GetRasterBand(1).WriteArray(newdata)
