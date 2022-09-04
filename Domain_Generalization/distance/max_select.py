#!/usr/bin/env python
# encoding: utf-8
'''
@Author: Yuqi
@Contact: www2048g@126.com
@File: max_select.py
@Time: 2022/9/3 14:23
'''
import torch


''' 将特征归一化到[0,1]'''
def data_normal(feature):
    d_min = feature.min()
    if d_min < 0:
        feature += torch.abs(d_min)
        d_min = feature.min()
    d_max = feature.max()
    dst = d_max - d_min
    normal_data = (feature - d_min).true_divide(dst)
    return normal_data

def max_distance_select(features):
    '''    ---------- 计算MMD ----------------   '''
    '''
        注意 在features里有大于1的数值，要处理一下
    '''
    from distance.MMD import mmd_rbf
    mul_feature = features.clone().detach()
    normal_features = data_normal(mul_feature)

    group_nums = 8

    chunk_features = torch.chunk(normal_features, group_nums, dim=0)
    index = []
    for i in range(group_nums):
        p = 0
        max_mmd = 0
        for j in range(group_nums):
            mmd = mmd_rbf(chunk_features[i], chunk_features[j])
            if max_mmd < mmd and j not in index:
                p = j
                max_mmd = mmd
            # print(i, j, ' ', mmd)
        index.append(p)
    # print(index)
    return index

