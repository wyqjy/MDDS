import argparse

import torch
#from IPython.core.debugger import set_trace
from torch import nn
#from torch.nn import functional as F
from data import data_helper
## from IPython.core.debugger import set_trace
from data.data_helper import available_datasets
from models import model_factory
from optimizer.optimizer_helper import get_optim_and_scheduler
from utils.Logger import Logger
import numpy as np
from models.resnet import resnet18, resnet50
import datetime
import time as time1
import os

import json
# 新加一个 tensorboard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./path/to/log')

CE = nn.CrossEntropyLoss()
RG = np.random.default_rng()

def get_args():
    parser = argparse.ArgumentParser(description="Script to launch jigsaw training",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source", choices=available_datasets, help="Source", nargs='+')
    parser.add_argument("--target", choices=available_datasets, help="Target")
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")  #受内存限制 改为32
    parser.add_argument("--image_size", type=int, default=222, help="Image size")
    # data aug stuff
    parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--random_horiz_flip", default=0.5, type=float, help="Chance of random horizontal flip")
    parser.add_argument("--jitter", default=0.4, type=float, help="Color jitter amount")
    parser.add_argument("--tile_random_grayscale", default=0.1, type=float, help="Chance of randomly greyscaling a tile")
    #
    parser.add_argument("--limit_source", default=None, type=int,
                        help="If set, it will limit the number of training samples")
    parser.add_argument("--limit_target", default=None, type=int,
                        help="If set, it will limit the number of testing samples")
    parser.add_argument("--learning_rate", "-l", type=float, default=.001, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=50, help="Number of epochs")  #默认20
    parser.add_argument("--n_classes", "-c", type=int, default=7, help="Number of classes")
    parser.add_argument("--network", choices=model_factory.nets_map.keys(), help="Which network to use", default="resnet18")
    parser.add_argument("--tf_logger", type=bool, default=True, help="If true will save tensorboard compatible logs")
    parser.add_argument("--val_size", type=float, default="0.1", help="Validation size (between 0 and 1)")
    parser.add_argument("--folder_name", default='test', help="Used by the logger to save logs")
    parser.add_argument("--bias_whole_image", default=0.9, type=float, help="If set, will bias the training procedure to show more often the whole image")
    parser.add_argument("--TTA", type=bool, default=False, help="Activate test time data augmentation")
    parser.add_argument("--classify_only_sane", default=False, type=bool, help="If true, the network will only try to classify the non scrambled images")
    parser.add_argument("--train_all", default=True, type=bool, help="If true, all network weights will be trained")
    parser.add_argument("--suffix", default="", help="Suffix for the logger")
    parser.add_argument("--nesterov", default=False, type=bool, help="Use nesterov")

    return parser.parse_args()


# Standard mix
def std_mix(x,indeces,ratio):
    #print(x.size(),' ',ratio.size(),' ',indeces.size())

    return ratio*x + (1.-ratio)*x[indeces]

# CE on mixed labels, represented as vectors
def manual_CE(predictions, labels):
    loss = -torch.mean(torch.sum(labels * torch.log_softmax(predictions,dim=1),dim=1))
    return loss

class CuMix:
    def __init__(self, configs, args):
        self.mixup_w = configs['mixup_img_w']
        self.mixup_feat_w = configs['mixup_feat_w']   # 特征级别的mixup的loss的权重比

        self.max_beta = configs['mixup_beta']
        self.mixup_beta = 0.0
        self.mixup_step = configs['mixup_step']

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

    def get_sample_mixup(self, data_bc):
        '''  目前先随机选  '''
        return torch.randperm(data_bc.shape[0])

    # Get ratio to perform mixup
    def get_ratio_mixup(self, domains):
        # print(domains.shape[0], ' ', self.mixup_beta)
        return torch.from_numpy(RG.beta(self.mixup_beta, self.mixup_beta, size=domains.shape[0])).float()

    def get_mixup_sample_and_ratio(self, data_bc, epoch):
        self.mixup_beta = min(self.max_beta, max(self.max_beta * (epoch) / self.mixup_step, 0.1))
        # if epoch>65:
        #     self.mixup_beta = 0.1
        return self.get_sample_mixup(data_bc), self.get_ratio_mixup(data_bc)

    # Get mixed inputs/labels
    def get_mixed_input_labels(self, input, labels, indeces, ratios, dims=2):
        if dims==4:
            return std_mix(input, indeces, ratios.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)), std_mix(labels, indeces, ratios.unsqueeze(-1))
        else:    # dims=2 表示传回的是两个东西， 一个是数据，一个是标签
            return std_mix(input, indeces, ratios.unsqueeze(-1)), std_mix(labels, indeces, ratios.unsqueeze(-1))




class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        if args.network == 'resnet18':
            model = resnet18(pretrained=True, classes=args.n_classes)
        elif args.network == 'resnet50':
            model = resnet50(pretrained=True, classes=args.n_classes)
        else:
            model = resnet18(pretrained=True, classes=args.n_classes)
        self.model = model.to(device)
        # print(self.model)
        self.source_loader, self.val_loader = data_helper.get_train_dataloader(args, patches=model.is_patch_based())
        self.target_loader = data_helper.get_val_dataloader(args, patches=model.is_patch_based())
        self.test_loaders = {"val": self.val_loader, "test": self.target_loader}
        self.len_dataloader = len(self.source_loader)
        print("Dataset size: train %d, val %d, test %d" % (
        len(self.source_loader.dataset), len(self.val_loader.dataset), len(self.target_loader.dataset)))
        self.optimizer, self.scheduler = get_optim_and_scheduler(model, args.epochs, args.learning_rate, args.train_all,
                                                                 nesterov=args.nesterov)
        self.n_classes = args.n_classes
        if args.target in args.source:
            self.target_id = args.source.index(args.target)
            print("Target in source: %d" % self.target_id)
            print(args.source)
        else:
            self.target_id = None

        self.dec_lr = 0.99

    def _do_epoch(self, epoch=None, CuMix_train=None):
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        print('-'*60)
        print('----- beta:', CuMix_train.mixup_beta)

        for it, ((data, jig_l, class_l), d_idx) in enumerate(self.source_loader):
            data, jig_l, class_l, d_idx = data.to(self.device), jig_l.to(self.device), class_l.to(self.device), d_idx.to(self.device)
            self.optimizer.zero_grad()

            data_flip = torch.flip(data, (3,)).detach().clone()  #按照维度对输入进行翻转,类别保持不变
            data = torch.cat((data, data_flip))   #原先的数据和翻转的数据进行拼接
            class_l = torch.cat((class_l, class_l))  #原先的类别和翻转的类别进行拼接

            class_logit, features = self.model(data, class_l, True, epoch, True, forward_feature=False)  #进行前向传播  forward   第三个参数True代表要进行RSC操作, 返回预测的类别  是否返回特征


            class_loss = criterion(class_logit, class_l)  #计算交叉熵损失
            _, cls_pred = class_logit.max(dim=1)  #获取最大的预测类别

            '''  ----------  CuMix   feature ----------'''
            one_hot_labels = CuMix_train.create_one_hot(class_l)
            mix_indeces, mix_ratios = CuMix_train.get_mixup_sample_and_ratio(data, epoch)
            mix_ratios = mix_ratios.to(self.device)
            mixup_features, mixup_labels = CuMix_train.get_mixed_input_labels(features, one_hot_labels, mix_indeces, mix_ratios)
            mixup_features_predictions = self.model(mixup_features, mixup_labels, False, epoch, False, forward_feature=True)  # 直接传进分类器层

            mixup_feature_loss = CuMix_train.mixup_criterion(mixup_features_predictions, mixup_labels)
            loss = CuMix_train.semantic_w*class_loss + CuMix_train.mixup_feat_w*mixup_feature_loss

            '''--------  CuMix  img --------'''
            mix_indeces, mix_ratios = CuMix_train.get_mixup_sample_and_ratio(data, epoch)
            mixup_inputs, mixup_labels = CuMix_train.get_mixed_input_labels(data, one_hot_labels, mix_indeces, mix_ratios.to(self.device), dims=4)
            mixup_img_predictions = self.model(mixup_inputs, mixup_labels, flag=False, return_features=False, forward_feature=False)
            mixup_img_loss = CuMix_train.mixup_criterion(mixup_img_predictions, mixup_labels)
            loss = loss + CuMix_train.mixup_w*mixup_img_loss


            # loss = class_loss

            loss.backward()
            self.optimizer.step()

            self.logger.log(it, len(self.source_loader),
                            {"class": class_loss.item()},
                            {"class": torch.sum(cls_pred == class_l.data).item(), }, data.shape[0])
            # writer.add_scalar('train/loss', class_loss.item(), it)       #损失值的图像

            del loss, class_loss, class_logit

        self.model.eval()  #固定 dropout 和 BN, 保证在验证和测试的时候，参数不变
        with torch.no_grad():
            for phase, loader in self.test_loaders.items():  #test_loaders里面是验证集和测试集（target）的数据
                total = len(loader.dataset)
                class_correct = self.do_test(loader)
                class_acc = float(class_correct) / total
                self.logger.log_test(phase, {"class": class_acc})
                self.results[phase][self.current_epoch] = class_acc

    def do_test(self, loader):  #返回有几个预测的准确   预测准确的个数
        class_correct = 0
        for it, ((data, nouse, class_l), _) in enumerate(loader):
            data, nouse, class_l = data.to(self.device), nouse.to(self.device), class_l.to(self.device)

            class_logit = self.model(data, class_l, False)
            _, cls_pred = class_logit.max(dim=1)

            class_correct += torch.sum(cls_pred == class_l.data)

        return class_correct

    def lr_Adjust(self, epoch):

        if epoch==88:
            for params in self.optimizer.param_groups:
                params['lr'] = 0.01
        # if epoch==13:
        #     for params in self.optimizer.param_groups:
        #         params['lr'] *= 0.1
        if epoch>30:
            for params in self.optimizer.param_groups:
                params['lr'] *= 0.999
                # if epoch==30:
                #     params['lr'] = 0.001
                # elif epoch<= 70:
                #     params['lr'] *= 0.83
                # else:
                #     params['lr'] *= self.dec_lr
                #     self.dec_lr -= 0.015



    def do_training(self):

        CuMix_config_file = "configs/dg.json"
        with open(CuMix_config_file) as json_file:
            CuMix_configs = json.load(json_file)

        CuMix_train = CuMix(CuMix_configs, self.args)

        self.logger = Logger(self.args, update_frequency=30)
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}
        for self.current_epoch in range(self.args.epochs):
            # print('-----',self.current_epoch)
            self.lr_Adjust(self.current_epoch)
            # self.scheduler.step()
            self.logger.new_epoch(self.scheduler.get_lr())
            for n, v in enumerate(self.scheduler.get_lr()):  #其实里面一直只有一个值
                writer.add_scalar('Learning rate', v, self.current_epoch)  #画学习率的图
            self._do_epoch(self.current_epoch, CuMix_train)

        writer.close() #新加的

        val_res = self.results["val"]

        test_res = self.results["test"]
        idx_best = val_res.argmax()
        idx_best_test = test_res.argmax()
        print("Best val %g, corresponding test %g val best epoch: %g------best test: %g, test best epoch: %g" % (val_res.max(), test_res[idx_best], idx_best+1, test_res.max(), idx_best_test+1))
        print('val acc\n', val_res, '\n\n')
        print('test acc\n', test_res)

        localtime = time1.localtime(time1.time())
        time = time1.strftime('%Y%m%d-%H.%M.%S', time1.localtime(time1.time()))
        da = str(datetime.datetime.today())
        filename = 'TXT/' + str(self.args.target) + '+RSC+CuMix_+epoch50' + '_'+ str(time) + '.txt'
        print(filename)
        file = open(filename, mode='w')
        file.write('best test' + str(test_res.max())+'   '+' local in'+str(idx_best_test+1)+'epoch'+'\n')
        file.write('Best val' + str(val_res.max)+'  '+'  local in'+str(idx_best+1)+'  corresponding test acc'+str(test_res[idx_best])+'\n\n')
        file.write('val acc\n'+str(val_res))
        file.write('\ntest acc\n'+str(test_res))


        self.logger.save_best(test_res[idx_best], test_res.max())  #存进来的是固定值
        return self.logger, self.model



def main():
    args = get_args()
    # args.source = ['art_painting', 'cartoon', 'sketch']
    # args.target = 'photo'
    # args.source = ['art_painting', 'cartoon', 'photo']
    # args.target = 'sketch'
    # args.source = ['art_painting', 'photo', 'sketch']
    # args.target = 'cartoon'
    # args.source = ['photo', 'cartoon', 'sketch']
    # args.target = 'art_painting'
    # --------------------------------------------
    print("Target domain: {}".format(args.target))
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(args, device)
    trainer.do_training()


if __name__ == "__main__":
    for i in range(1):
        torch.backends.cudnn.benchmark = True  #设置为True，会使得cuDNN来衡量自己库里面的多个卷积算法的速度，然后选择其中最快的那个卷积算法
        main()
