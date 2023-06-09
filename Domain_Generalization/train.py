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
from models.alexnet import alexnet
from models.mixup import Mixup
import datetime
import time as time1
import os

from os.path import join
import json

from distance.max_select import max_distance_select
# 新加一个 tensorboard
from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('./path/to/log')



def get_args():
    parser = argparse.ArgumentParser(description="Script to launch jigsaw training",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source", choices=available_datasets, help="Source", nargs='+')
    parser.add_argument("--target", choices=available_datasets, help="Target")
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")  #受内存限制 改为32
    parser.add_argument("--image_size", type=int, default=222, help="Image size")  # digits32(Lenet)  vlcs (alex) 224  resnet222
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
    parser.add_argument("--epochs", "-e", type=int, default=30, help="Number of epochs")  #默认20
    parser.add_argument("--n_classes", "-c", type=int, default=7, help="Number of classes")
    parser.add_argument("--network", choices=model_factory.nets_map.keys(), help="Which network to use", default="resnet50")
    parser.add_argument("--tf_logger", type=bool, default=True, help="If true will save tensorboard compatible logs")
    parser.add_argument("--val_size", type=float, default="0.1", help="Validation size (between 0 and 1)")
    parser.add_argument("--folder_name", default='test', help="Used by the logger to save logs")
    parser.add_argument("--bias_whole_image", default=0.9, type=float, help="If set, will bias the training procedure to show more often the whole image")
    parser.add_argument("--TTA", type=bool, default=False, help="Activate test time data augmentation")
    parser.add_argument("--classify_only_sane", default=False, type=bool, help="If true, the network will only try to classify the non scrambled images")
    parser.add_argument("--train_all", default=True, type=bool, help="If true, all network weights will be trained")
    parser.add_argument("--suffix", default="", help="Suffix for the logger")
    parser.add_argument("--nesterov", default=False, type=bool, help="Use nesterov")

    parser.add_argument("--no_train", default=False, type=bool, help="only test")
    parser.add_argument("--dataset", default='pacs', help="dataset")
    parser.add_argument("--seed", type=int, default=0, help="seed")

    return parser.parse_args()


class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        if args.network == 'resnet18':
            model = resnet18(pretrained=True, classes=args.n_classes)
        elif args.network == 'resnet50':
            model = resnet50(pretrained=True, classes=args.n_classes)
        elif args.network == 'alexnet':
            model = alexnet(pretrained=True, classes=args.n_classes)
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

        torch.set_num_threads(4)
        self.dec_lr = 0.99

    def _do_epoch(self, epoch=None, Mix_train=None):
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        print('-'*60)
        print('----- beta:', Mix_train.mixup_beta)

        for it, ((data, jig_l, class_l), d_idx) in enumerate(self.source_loader):
            data, jig_l, class_l, d_idx = data.to(self.device), jig_l.to(self.device), class_l.to(self.device), d_idx.to(self.device)
            self.optimizer.zero_grad()

            data_flip = torch.flip(data, (3,)).detach().clone()  #按照维度对输入进行翻转,类别保持不变
            data = torch.cat((data, data_flip))   #原先的数据和翻转的数据进行拼接
            class_l = torch.cat((class_l, class_l))  #原先的类别和翻转的类别进行拼接
            d_idx = torch.cat((d_idx, d_idx))

            class_logit, features = self.model(data, class_l, True, epoch, True, forward_feature=False)  #进行前向传播  forward   第三个参数True代表要进行RSC操作, 返回预测的类别  是否返回特征


            class_loss = criterion(class_logit, class_l)  #计算交叉熵损失
            _, cls_pred = class_logit.max(dim=1)  #获取最大的预测类别

            # torch.save(features, 'tensor\\origin-features')
            # torch.save(d_idx, 'tensor\\origin-domain')

            '''  ----------  CuMix   feature ----------'''

            max_dis_index = max_distance_select(features=features)

            one_hot_labels = Mix_train.create_one_hot(class_l)
            mix_indeces, mix_ratios = Mix_train.get_mixup_sample_and_ratio(d_idx, epoch, random=False, max_dis_index=max_dis_index)
            mix_ratios = mix_ratios.to(self.device)
            mixup_features, mixup_labels = Mix_train.get_mixed_input_labels(features, one_hot_labels, mix_indeces, mix_ratios)

            # torch.save(mixup_features, 'tensor\\2mix-features')
            # _, l2 = mixup_labels.max(dim=1)
            # torch.save(l2, 'tensor\\2mix-label')

            # 四样本
            max_dis_index = max_distance_select(features=mixup_features)
            mix_indeces, mix_ratios = Mix_train.get_mixup_sample_and_ratio(d_idx, epoch, random=False, max_dis_index=max_dis_index)
            mix_ratios = mix_ratios.to(self.device)
            mixup_features, mixup_labels = Mix_train.get_mixed_input_labels(mixup_features, mixup_labels, mix_indeces, mix_ratios)

            # torch.save(mixup_features, 'tensor\\4mix-features')
            # _, l4 = mixup_labels.max(dim=1)
            # torch.save(l4, 'tensor\\4mix-label')

            mixup_features_predictions = self.model(mixup_features, mixup_labels, False, epoch, False, forward_feature=True)  # 直接传进分类器层

            mixup_feature_loss = Mix_train.mixup_criterion(mixup_features_predictions, mixup_labels)
            loss = Mix_train.semantic_w*class_loss + Mix_train.mixup_feat_w*mixup_feature_loss

            '''--------  Mix  img --------'''
            # mix_indeces, mix_ratios = Mix_train.get_mixup_sample_and_ratio(d_idx, epoch)
            # mixup_inputs, mixup_labels = Mix_train.get_mixed_input_labels(data, one_hot_labels, mix_indeces, mix_ratios.to(self.device), dims=4)
            # mixup_img_predictions = self.model(mixup_inputs, mixup_labels, flag=False, return_features=False, forward_feature=False)
            # mixup_img_loss = Mix_train.mixup_criterion(mixup_img_predictions, mixup_labels)
            # loss = loss + Mix_train.mixup_w*mixup_img_loss


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
                self.logger.log_test(phase, {"class": class_acc}, self.model, self.args)
                self.results[phase][self.current_epoch] = class_acc

    def do_test(self, loader):  #返回有几个预测的准确   预测准确的个数
        class_correct = 0
        for it, ((data, nouse, class_l), _) in enumerate(loader):
            data, nouse, class_l = data.to(self.device), nouse.to(self.device), class_l.to(self.device)

            class_logit = self.model(data, class_l, False)
            _, cls_pred = class_logit.max(dim=1)

            class_correct += torch.sum(cls_pred == class_l.data)

        return class_correct

    # def lr_Adjust(self, epoch):
    #
    #     if epoch==10:
    #         for params in self.optimizer.param_groups:
    #             params['lr'] = 0.01
    #     # if epoch > 20 and epoch < 35:
    #     #     for params in self.optimizer.param_groups:
    #     #         params['lr'] *= 0.999
    #     if epoch==30:
    #         for params in self.optimizer.param_groups:
    #             params['lr'] = 0.001
    #     if epoch>30:
    #         for params in self.optimizer.param_groups:
    #             params['lr'] *= 0.99
    #             # if epoch==30:
    #             #     params['lr'] = 0.001
    #             # elif epoch<= 70:
    #             #     params['lr'] *= 0.83
    #             # else:
    #             #     params['lr'] *= self.dec_lr
    #             #     self.dec_lr -= 0.015

    def only_test(self):
        '''  用训练好的模型测试，不训练  '''
        print("Only Test")
        model_dir = join("output", self.args.target, "0.956287_resnet18.pth")
        self.model = torch.load(model_dir)
        total = len(self.target_loader.dataset)
        self.model.eval()
        with torch.no_grad():
            class_correct = self.do_test(self.target_loader)
            class_acc = float(class_correct) / total
            print(class_acc, ' ', total)
        print("Test on", self.args.target, " accury", str(class_acc*100))

    def do_training(self):

        Mix_config_file = "configs/dg.json"
        with open(Mix_config_file) as json_file:
            Mix_configs = json.load(json_file)

        Mix_train = Mixup(Mix_configs, self.args)
        # print(CuMix_train.semantic_w)
        # print(CuMix_train.mixup_feat_w)
        # print(CuMix_train.mixup_w)

        self.logger = Logger(self.args, update_frequency=30)
        self.logger.record_default(Mix_configs)
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}

        for self.current_epoch in range(self.args.epochs):
            # print('-----',self.current_epoch)
            # self.lr_Adjust(self.current_epoch)
            self.scheduler.step()
            self.logger.new_epoch(self.scheduler.get_lr())
            # for n, v in enumerate(self.scheduler.get_lr()):  #其实里面一直只有一个值
            #     writer.add_scalar('Learning rate', v, self.current_epoch)  #画学习率的图
            self._do_epoch(self.current_epoch, Mix_train)

        # writer.close() #新加的

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
        filefolder = "output/" + str(self.args.dataset) + '/' + 'seed'+str(self.args.seed) + '/TXT'
        if not (os.path.exists(filefolder)):
            os.makedirs(filefolder)
        filename = filefolder + '/' + str(self.args.target) + '+epochs' + str(self.args.epochs) + '_' + str(time) + '.txt'
        print(filename)
        file = open(filename, mode='w')
        record_best_test = 'best test' + str(test_res.max())+'   '+' local in'+str(idx_best_test+1)+'epoch'
        self.logger.record_logs(record_best_test)
        file.write(record_best_test + '\n')
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
    # args.source = ['art', 'clipart', 'product']
    # args.target = 'real_world'
    # args.source = ['art', 'clipart', 'real_world']
    # args.target = 'product'
    # args.source = ['art', 'real_world', 'product']
    # args.target = 'clipart'
    # args.source = ['real_world', 'clipart', 'product']
    # args.target = 'art'
    # ---------------------------------------------
    # args.source = ["LABELME", "PASCAL", "SUN"]
    # args.target = "CALTECH"
    # args.source = ["CALTECH", "PASCAL", "SUN"]
    # args.target = "LABELME"
    # args.source = ["CALTECH", "LABELME", "SUN"]
    # args.target = "PASCAL"
    # args.source = ["CALTECH", "LABELME", "PASCAL"]
    # args.target = "SUN"
    # ---------------------------------------------
    # args.source = ['quickdraw', 'sketch', 'real', 'infograph', 'painting']
    # args.target = 'clipart'
    # args.source = ['clipart','sketch', 'real', 'infograph', 'painting']
    # args.target = 'quickdraw'
    # args.source = ['clipart', 'quickdraw', 'real', 'infograph', 'painting']
    # args.target = 'sketch'
    # args.source = ['clipart', 'quickdraw', 'sketch', 'infograph', 'painting']
    # args.target = 'real'
    # args.source = ['clipart', 'quickdraw', 'sketch', 'real', 'painting']
    # args.target = 'infograph'
    # args.source = ['clipart', 'quickdraw', 'sketch', 'real', 'infograph']
    # args.target = 'painting'
    # ---------------------------------------------
    # args.source = ["L38", "L43", "L46"]
    # args.target = "L100"
    # args.source = ["L100", "L43", "L46"]
    # args.target = "L38"
    # args.source = ["L100", "L38", "L46"]
    # args.target = "L43"
    # args.source = ["L100", "L38", "L43"]
    # args.target = "L46"

    print("Target domain: {}".format(args.target))
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dataset=='pacs':
        args.n_classes = 7
    elif args.dataset=='officehome':
        args.n_classes = 65
    elif args.dataset == 'vlcs':
        args.n_classes = 5
    elif args.dataset == 'DomainNet':
        args.n_classes = 345
    elif args.dataset == 'TerraIncognita':
        args.n_classes = 10

    trainer = Trainer(args, device)
    if not args.no_train:
        trainer.do_training()
    else:
        trainer.only_test()


if __name__ == "__main__":
    for i in range(1):
        torch.backends.cudnn.benchmark = True  #设置为True，会使得cuDNN来衡量自己库里面的多个卷积算法的速度，然后选择其中最快的那个卷积算法
        main()
