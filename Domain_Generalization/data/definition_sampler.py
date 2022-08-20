import numpy as np
import torch
import os
import random
import torch.utils.data as data


'''
    每个batch的每个域的样本数量相等
    虽然是用concat_dataset，但还是返回正常的索引序列
'''
class DistributedBalancedSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, domains_per_batch, samplers_per_domain, shuffer=True):
        self.dataset = dataset
        self.datasets = self.dataset.datasets  # 对于每个域标签建立的字典，不过每个的索引都是从0开始的
        self.dpb = domains_per_batch    # 一个batch里有几个领域
        self.dbs = samplers_per_domain  # 一个batch中的一个domain有几个样本 22
        self.dict_domain = {}
        # self.domain_ids = np.array(len(self.dataset.datasets))
        self.n_doms = len(self.datasets)
        self.cumulative_sizes = {}

        self.indeces = {}    # 记录某一个领域已经分配出去的样本

        for d in range(self.n_doms):
            self.dict_domain[d] = []
            self.indeces[d]=0

        idk = 0
        for d in range(self.n_doms):
            for idx in range(len(self.datasets[d])):
                self.dict_domain[d].append(idk)
                idk += 1
            self.cumulative_sizes[d] = len(self.datasets[d])

        if shuffer:
            for idx in range(self.n_doms):
                random.shuffle(self.dict_domain[idx])
        self.rep = {}  # 计算一下重复利用样本的个数
        for d in range(self.n_doms):
            self.rep[d] = 0
        self.samples = torch.LongTensor(self._get_samples())



    def _sampling(self, d_idx, n):
        if self.indeces[d_idx] + n >= self.cumulative_sizes[d_idx]:  # 剩下的样本数量不够再分配一次的
            # 随机找n个
            self.indeces[d_idx] = self.indeces[d_idx] + n
            self.rep[d_idx] += n
            return random.sample(self.dict_domain[d_idx], n)

        self.indeces[d_idx] = self.indeces[d_idx] + n
        return self.dict_domain[d_idx][self.indeces[d_idx] - n : self.indeces[d_idx]]

    def _get_samples(self):
        sIdx = []
        useless = {}
        for d in range(self.n_doms):
            useless[d] = 0
        while 1:
            for d in range(self.n_doms):    # 一个batch
                sIdx += self._sampling(d, self.dbs)
                if self.indeces[d] >= self.cumulative_sizes[d]:
                    useless[d] = 1
            sum = 0
            for d in range(self.n_doms):
                sum += useless[d]
            if sum == self.n_doms:
                break
        return np.array(sIdx)

    def __iter__(self):
        indices = list(range(len(self.samples)))
        return iter(self.samples[indices])

    def __len__(self):
        return len(self.samples)
