#!/usr/bin/env python
# encoding: utf-8
'''
@Author: Yuqi
@Contact: www2048g@126.com
@File: Beta.py
@Time: 2022/9/16 14:40
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import math


def vis(x, y):
    plt.plot(x, y)
    plt.show()


x = []
y = []
max_beta = 0.6
mixup_step = 10
for epoch in range(30):
    ''' 原来的线性'''
    x.append(epoch)
    mixup_beta = min(max_beta, max(max_beta * (epoch) / mixup_step, 0.1))
    y.append(mixup_beta)
    # print(epoch, ' : ', mixup_beta)


def fun(a):
    x_pow = []
    y_pow = []
    for epoch in range(30):
        ''' 指数函数 左侧部分<1  '''
        pow_beta = min(max_beta, math.pow(a, epoch-20))  # 加不加最小值0.1待定
        y_pow.append(pow_beta)
        print(epoch, ':', pow_beta, ' '*10, math.pow(a, epoch-20))
    return y_pow

vis(x, y)
a=1.1  # 从1.1-1.5中取值
for i in range(1):
    y_pow = fun(a)
    a+=0.1
    vis(x, y_pow)