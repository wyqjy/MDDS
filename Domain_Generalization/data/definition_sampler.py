import copy
import numpy as np
import torch
import os
import random
import torch.utils.data as data
from collections import defaultdict

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
        print()

    def _sampling(self, d_idx, n):
        if self.indeces[d_idx] + n >= self.cumulative_sizes[d_idx]:  # 剩下的样本数量不够再分配一次的
            # 随机找n个
            self.indeces[d_idx] = self.indeces[d_idx] + n
            self.rep[d_idx] += n
            return random.sample(self.dict_domain[d_idx], n)
            ''' 降低重复率后效果变差了 因为数据量少了'''
            # d = -1   # 找剩余样本最多的进行分配
            # cha = 0
            # for i in range(self.n_doms):
            #     c = self.cumulative_sizes[i] - self.indeces[i]
            #     if c > n and c > cha:
            #         cha = c
            #         d = i
            # if d > -1:
            #     d_idx = d
            # else:   #每个域剩下的样本都不足以构成一组
            #     self.rep[d_idx] += n
            #     return random.sample(self.dict_domain[d_idx], n)

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
                if self.indeces[d] + self.dbs >= self.cumulative_sizes[d]:  # 不足构成一个batch
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

class Base_Sampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, shuffle=True):
        self.dataset = dataset.datasets  # 对于每个域标签建立的字典，不过每个的索引都是从0开始的
        self.dict_domain = {}     # 按领域划分的字典
        self.dict_domain_and_label = defaultdict(lambda: defaultdict(list))   # 按领域 标签 划分的字典
        self.n_doms = len(self.dataset)          # 数据集中源域的数量
        self.cumulative_domain_sizes = {}         # 每个领域的样本数量
        self.indeces = {}  # 记录某一个领域已经分配出去的样本
        for d in range(self.n_doms):
            self.dict_domain[d] = []
            self.indeces[d] = 0
            self.cumulative_domain_sizes[d] = len(self.dataset[d])

        idk = 0
        for d in range(self.n_doms):                          # 生成字典
            for idx in range(len(self.dataset[d])):
                label = self.dataset[d].labels[idx]
                self.dict_domain[d].append(idk)
                self.dict_domain_and_label[d][label].append(idk)
                idk += 1

        self.classes = len(self.dict_domain_and_label[0])   # 每个领域的类别数
        if shuffle:                                         # 对索引下标再随机
            for idx in range(self.n_doms):
                random.shuffle(self.dict_domain[idx])
                for label_idx in range(self.classes+1):
                    random.shuffle(self.dict_domain_and_label[idx][label_idx])

    def _sampling(self):
        pass

    def _get_samples(self):
        pass

    def __iter__(self):
        pass

    def __len__(self):
        pass

class split_domain_Sampler(Base_Sampler):
    def __init__(self, dataset, batchsize, n_domains, repeat=True):
        super().__init__(dataset)
        self._name = '按领域划分'
        self.batchsize = batchsize
        self.n_doms = n_domains    # 一个batch 要包含几个域
        self.per_domain = batchsize // n_domains  #一个batch中一个域包含几个样本
        self.domains = list(self.dict_domain.keys())
        self.repeat = repeat  # 是直接抛弃还是重复利用
        if self.per_domain*self.n_doms != self.batchsize:
            raise ValueError(
                "batchsize must be {} Integer times".format(self.n_doms)
            )

        self.rep = {}  # 计算一下重复利用样本的个数
        for d in range(self.n_doms):
            self.rep[d] = 0

        if self.n_doms == 3:
            self.samples = torch.LongTensor(self._get_samples())
        else:
            self.samples = torch.LongTensor(self._mixstyle_Samples())

        print()

    def _sampling(self, d_idx, n):   # 为d_idx这个域分配出去
        if self.indeces[d_idx] + n >= self.cumulative_domain_sizes[d_idx]:  # 剩下的样本数量不够再分配一次的
            # 随机找n个 可重复选（选择之前用过的）
            if self.repeat:
                self.indeces[d_idx] = self.indeces[d_idx] + n
                self.rep[d_idx] += n
                return random.sample(self.dict_domain[d_idx], n)   # 随机找n个，使用过的
            else:
                ''' 降低重复率后效果变差了 '''
                d = -1   # 找剩余样本最多的进行分配
                cha = 0
                for i in range(self.n_doms):
                    c = self.cumulative_domain_sizes[i] - self.indeces[i]
                    if c > n and c > cha:
                        cha = c
                        d = i
                if d > -1:
                    d_idx = d
                else:   #每个域剩下的样本都不足以构成一组
                    self.rep[d_idx] += n
                    return random.sample(self.dict_domain[d_idx], n)

        self.indeces[d_idx] = self.indeces[d_idx] + n     # 对应领域 分配出去的样本数量加n
        return self.dict_domain[d_idx][self.indeces[d_idx] - n: self.indeces[d_idx]]

    def _get_samples(self):
        sIdx = []
        useless = {}   # 各个领域已经用了多少的样本了
        for d in range(self.n_doms):
            useless[d] = 0

        while 1:
            for d in range(self.n_doms):  # 一个batch
                sIdx += self._sampling(d, self.per_domain)
                if self.indeces[d] + self.per_domain >= self.cumulative_domain_sizes[d]:  # 不足构成一个batch，用光了一个域的数据
                    useless[d] = 1
            sum = 0
            for d in range(self.n_doms):
                sum += useless[d]
            if sum == self.n_doms:
                break
        return np.array(sIdx)

    def _mixstyle_Samples(self):
        domain_dict = copy.deepcopy(self.dict_domain)
        final_idxs = []
        stop_sampling = False
        n_img_per_domain = self.batchsize // self.n_doms

        while not stop_sampling:
            selected_domains = random.sample(self.domains, self.n_doms)

            for domain in selected_domains:
                idxs = domain_dict[domain]
                selected_idxs = random.sample(idxs, n_img_per_domain)
                final_idxs.extend(selected_idxs)

                for idx in selected_idxs:    # 移除选中的，不会二次重复
                    domain_dict[domain].remove(idx)

                remaining = len(domain_dict[domain])
                if remaining < n_img_per_domain:
                    stop_sampling = True

        return np.array(final_idxs)


    def __iter__(self):
        indices = list(range(len(self.samples)))
        return iter(self.samples[indices])

    def __len__(self):
        return len(self.samples)
