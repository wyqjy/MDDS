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
import torch

RG = np.random.default_rng()

def vis(x, y):
    plt.plot(x, y)
    plt.show()


x = []
y = []
max_beta = 0.6
mixup_step = 10
def orgin():
    for epoch in range(30):
        ''' 原来的线性'''
        x.append(epoch)
        mixup_beta = min(max_beta, max(max_beta * (epoch) / mixup_step, 0.1))
        y.append(mixup_beta)
        # print(epoch, ' : ', mixup_beta)


def fun(max_beta, a, x):
    x_pow = []
    y_pow = []
    for epoch in range(30):
        ''' 指数函数 左侧部分<1  '''
        pow_beta = min(max_beta, math.pow(a, epoch-x))  # 加不加最小值0.1待定
        y_pow.append(pow_beta)
        # print(epoch, ':', pow_beta, ' '*10, math.pow(a, epoch-20))
    return y_pow

def lamuda():
    max_beta = 0.1
    print(RG.beta(max_beta, max_beta, size=10))
    print(torch.from_numpy(RG.beta(max_beta, max_beta, size=10)).float())

orgin()
vis(x, y)
fig, ax = plt.subplots()
a=1.2  # 从1.1-1.5中取值
max_beta = 0.6
xx = 10
for i in range(4):
    y_pow = fun(max_beta, a, xx)
    ax.plot(x, y_pow, label='x='+str(xx))
    ax.legend()
    xx += 5
    xx = round(xx, 1)
    print(xx)
    # vis(x, y_pow)
ax.set_xlabel('epoch')
ax.set_ylabel('alpha')
plt.savefig('Beta 参数变化 x.jpg')
plt.show()
# lamuda()


def Be():
    import numpy as np
    from scipy.stats import beta
    import matplotlib.pyplot as plot

    # 设置 plot 支持中文
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=14)

    # 定义一组alpha 跟 beta值
    # alpha_beta_values = [[0.5, 0.5], [5, 1], [1, 3], [2, 2], [2, 5]]
    alpha_beta_values = [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.5, 0.5]]
    linestyles = []

    # 定义 x 值
    x = np.linspace(0, 1, 1002)[1:-1]
    for alpha_beta_value in alpha_beta_values:
        print(alpha_beta_value)
        dist = beta(alpha_beta_value[0], alpha_beta_value[1])
        dist_y = dist.pdf(x)
        # 添加图例
        # plot.legend('alpha=')
        # 创建 beta 曲线
        plot.plot(x, dist_y, label=r'$\alpha=%.1f,\ \beta=%.1f$' % (alpha_beta_value[0], alpha_beta_value[1]))

    # 设置标题
    plot.title(u'Beta分布', fontproperties=font)
    # 设置 x,y 轴取值范围
    plot.xlim(0, 1)
    plot.ylim(0, 2.5)
    plot.legend()
    plt.savefig('Beta分布.jpg')
    plot.show()

# Be()