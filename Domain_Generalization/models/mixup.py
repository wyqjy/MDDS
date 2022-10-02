import torch
import numpy as np
from torch import nn

CE = nn.CrossEntropyLoss()
RG = np.random.default_rng()

# Standard mix
def std_mix(x,indeces,ratio):
    #print(x.size(),' ',ratio.size(),' ',indeces.size())

    return ratio*x + (1.-ratio)*x[indeces]

# CE on mixed labels, represented as vectors
def manual_CE(predictions, labels):
    loss = -torch.mean(torch.sum(labels * torch.log_softmax(predictions,dim=1),dim=1))
    return loss

def swap(xs, a, b):
    xs[a], xs[b] = xs[b], xs[a]

def derange(xs):
    x_new = [] + xs
    for a in range(1, len(x_new)):
        b = RG.choice(range(0, a))
        swap(x_new, a, b)
    return x_new

class Mixup:
    def __init__(self, configs, args):
        self.mixup_w = configs['mixup_img_w']
        self.mixup_feat_w = configs['mixup_feat_w']   # 特征级别的mixup的loss的权重比

        self.max_beta = configs['mixup_beta']
        self.mixup_beta = 0.0
        self.mixup_step = configs['mixup_step']
        self.mixup_domain = 0

        self.step = configs['step']
        # self.batch_size = configs['batch_size']
        # self.lr = configs['lr']
        # self.nesterov = configs['nesterov']
        # self.decay = configs['weight_decay']
        # self.freeze_bn = configs['freeze_bn']

        # input_dim = configs['input_dim']
        self.semantic_w = configs['semantic_w']

        self.seen_classes = args.n_classes

        self.mixup_criterion = manual_CE
        self.device = "cuda"


    # Create one hot labels
    def create_one_hot(self, y):
        y_onehot = torch.LongTensor(y.size(0), self.seen_classes).to(self.device) # .size(0)
        y_onehot.zero_()
        y_onehot.scatter_(1, y.view(-1, 1), 1)
        return y_onehot

    def get_random_sample_mixup(self, domains):
        '''  随机选   '''
        res = torch.randperm(domains.shape[0])
        return res

    def get_sample_mixup(self, domains, max_dis_index=None):

        group_nums = 8
        min_group = int(domains.shape[0]/group_nums)
        index = torch.IntTensor()
        for i in range(group_nums):
            min_index = torch.randperm(int(min_group))
            index = torch.cat((index, min_index+max_dis_index[i]*int(min_group)), dim=0)
        # print(index)
        return index



        ''' 每个batch里都有相等的域样本数 不好用'''
        # doms = list(range(len(torch.unique(domains))))  # [0,1,2]    挑出独立不重复的元素
        # c = domains.size(0)
        # bs1 = domains.size(0) // len(doms)  # 一个batch里一个域要包含的样本数量
        # bs = bs1 // 2
        # selected = derange(doms)  # 重新排列领域标号
        # permuted_across_dom = torch.cat([(torch.randperm(bs) + selected[i] * bs) for i in range(len(doms))])
        # permuted_within_dom = torch.cat([(torch.randperm(bs) + i * bs) for i in range(len(doms))])
        # ratio_within_dom = torch.from_numpy(RG.binomial(1, self.mixup_domain, size=domains.size(0)//2))
        # indeces = ratio_within_dom * permuted_within_dom + (1. - ratio_within_dom) * permuted_across_dom
        #
        # indeces_la = indeces.add(domains.size(0)//2)
        # indeces = torch.cat((indeces, indeces_la))
        # return indeces.long()


    # Get ratio to perform mixup
    def get_ratio_mixup(self, domains):
        # print(domains.shape[0], ' ', self.mixup_beta)
        return torch.from_numpy(RG.beta(self.mixup_beta, self.mixup_beta, size=domains.shape[0])).float()

    def get_mixup_sample_and_ratio(self, data_bc, epoch, random=False, max_dis_index=None):
        # self.mixup_beta = min(self.max_beta, max(self.max_beta * (epoch) / self.mixup_step, 0.1))
        import math
        self.mixup_beta = min(self.max_beta, math.pow(1.1, epoch - 20))

        self.mixup_domain = min(1.0, max((self.mixup_step * 2. - epoch) / self.mixup_step, 0.0))
        # if epoch>65:
        #     self.mixup_beta = 0.1
        if random:
            return self.get_random_sample_mixup(data_bc), self.get_ratio_mixup(data_bc)
        return self.get_sample_mixup(data_bc, max_dis_index=max_dis_index), self.get_ratio_mixup(data_bc)

    # Get mixed inputs/labels
    def get_mixed_input_labels(self, input, labels, indeces, ratios, dims=2):
        if dims==4:
            return std_mix(input, indeces, ratios.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)), std_mix(labels, indeces, ratios.unsqueeze(-1))
        else:    # dims=2 表示传回的是两个东西， 一个是数据，一个是标签
            return std_mix(input, indeces, ratios.unsqueeze(-1)), std_mix(labels, indeces, ratios.unsqueeze(-1))
