#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, re
import pandas as pd
import numpy as np

area = pd.read_csv('E:\\城市与区域生态\\保护区规划Marxan\\示例数据\\示例数据\\area.csv', header=0)
# print(area)
print(area['area'][0], area['area'][1053])
output = []  # 输出文件集合
choosen_area = []  # 存放输出文件中每一个被选单元的面积
sum_choosen_area = []  # 存放每一个输出文件面积之和的占比
for root, ds, fs in os.walk(r'E:\城市与区域生态\保护区规划Marxan\示例数据\示例数据\T1234\BLM\50\0.00005\output'):
    for f in fs:
        if re.match(r'^output_r0.*', f):
            fullname = os.path.join(root, f)
            output.append(fullname)

for o in output:
    choosen = pd.read_csv(o, header=0)
    for i in range(1054):
        if choosen['SOLUTION'][i] == 1:
            choosen_area.append(area['area'][i])
    sum_choosen_area.append(sum(choosen_area))
    choosen_area.clear()
sum_choosen_area = np.array(sum_choosen_area)
sum_choosen_area = sum_choosen_area / 10466.88703 * 100
print('mean:', np.mean(sum_choosen_area))
print('std:', np.std(sum_choosen_area))
