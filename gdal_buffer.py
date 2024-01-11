#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path
from osgeo import ogr


def buffer(inShp, fname):
    """
    :param inShp: 输入的矢量路径
    :param fname: 输出的矢量路径
    :return:
    """
    ogr.UseExceptions()
    in_ds = ogr.Open(inShp)
    in_lyr = in_ds.GetLayer()
    in_feature = in_lyr.GetNextFeature()
    geometry = in_feature.GetGeometryRef()
    # 创建输出Buffer文件

    # 遍历原始的Shapefile文件给每个Geometry做Buffer操作
    for i in range(10):
        driver = ogr.GetDriverByName('ESRI Shapefile')
        # 新建DataSource，Layer
        fnameOut = fname + "{0}".format(i * 500 + 500)
        out_ds = driver.CreateDataSource("E:\\城市与区域生态\\大熊猫和竹\\道路对大熊猫栖息地的影响\\道路距离分析\\15缓冲区1229FLAT")
        out_lyr = out_ds.CreateLayer(fnameOut, in_lyr.GetSpatialRef(), ogr.wkbPolygon)
        def_feature = out_lyr.GetLayerDefn()
        buffer = geometry.Buffer(500 * i + 500)
        out_feature = ogr.Feature(def_feature)
        out_feature.SetGeometry(buffer)
        out_lyr.CreateFeature(out_feature)
        out_feature = None
    out_ds.FlushCache()
    del in_ds, out_ds


if __name__ == '__main__':
    inShp = 'E:\\城市与区域生态\\大熊猫和竹\\道路对大熊猫栖息地的影响\\道路数据\\road_lzc\\1229\\研究区高速国道省道15非隧道PCS.shp'
    fname = '2015_'
    buffer(inShp, fname)
