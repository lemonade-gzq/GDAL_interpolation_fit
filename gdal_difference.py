#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, shutil, sys
from osgeo import ogr, gdal, gdal_array, ogr, osr
import math, numpy

"""基于OSGEO的两个面状矢量文件空间操作：相交、交集取反、相减和合并"""


def shapefile_geometric_relations_operations(_input_shapefile_1, _input_shapefile_2, result_shapfile, action=1):
    '''
    _input_shapefile_1第一个矢量文件的绝对路径
    _input_shapefile_2第二个矢量文件的绝对路径
    result_shapfile输出矢量文件绝对路径
    action=1表示取两个矢量的交集intersection
          =2表示取两个矢量交集的反集sysdifference
          =3表示两个矢量相减difference
          =4表示取两个矢量的并集union
    这个函数对输入的矢量文件没有坐标系的要求，输出的矢量文件和输入的矢量文件坐标系相同
    '''
    for i in range(84):
        input_shapefile_1 = _input_shapefile_1 + "{}.shp".format(i * 500 + 1000)
        input_shapefile_2 = _input_shapefile_2 + "{}.shp".format(i * 500 + 500)
        out_shapefile = "2000XD_{0}_{1}条带".format(i * 500 + 500, i * 500 + 1000)
        _ds_1, _ds_2 = ogr.Open(input_shapefile_1), ogr.Open(input_shapefile_2)
        _lyr_1, _lyr_2 = _ds_1.GetLayer(), _ds_2.GetLayer()
        shpdriver = ogr.GetDriverByName('ESRI Shapefile')
        # if os.path.exists(result_shapfile):
        #     shpdriver.DeleteDataSource(result_shapfile)

        outDifference = shpdriver.CreateDataSource(result_shapfile)
        outDifferenceLyr = outDifference.CreateLayer(out_shapefile, _lyr_1.GetSpatialRef(), geom_type=ogr.wkbPolygon)
        outDifferenceFeatDefn = outDifferenceLyr.GetLayerDefn()

        for _feat_1, _feat_2 in zip(_lyr_1, _lyr_2):
            _geomtry_1, _geomtry_2 = _feat_1.GetGeometryRef(), _feat_2.GetGeometryRef()

        flag = 0
        # flag反馈两个矢量文件是否相交的表示，通过return返回，1表示相交，0表示不想交
        if action == 1:
            _result_polygon = _geomtry_1.Intersection(_geomtry_2)
            if _result_polygon == True:
                Flag = 1
        elif action == 2:
            _result_polygon = _geomtry_1.SymDifference(_geomtry_2)
        elif action == 3:
            _result_polygon = _geomtry_1.Difference(_geomtry_2)
        elif action == 4:
            _result_polygon = _geomtry_1.Union(_geomtry_2)

        _outFeature = ogr.Feature(outDifferenceFeatDefn)
        _outFeature.SetGeometry(_result_polygon)
        outDifferenceLyr.CreateFeature(_outFeature)
        _outFeature = None
        # return flag


if __name__ == '__main__':
    _input_shapefile_1 = 'E:\\城市与区域生态\\大熊猫和竹\\道路对大熊猫栖息地的影响\\道路距离分析\\00县道缓冲区1229all\\2000XD_'
    _input_shapefile_2 = 'E:\\城市与区域生态\\大熊猫和竹\\道路对大熊猫栖息地的影响\\道路距离分析\\00县道缓冲区1229all\\2000XD_'
    result_shapfile = 'E:\\城市与区域生态\\大熊猫和竹\\道路对大熊猫栖息地的影响\\道路距离分析\\00县道条带1229all'
    shapefile_geometric_relations_operations(_input_shapefile_1, _input_shapefile_2, result_shapfile, action=3)