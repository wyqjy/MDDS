import os.path
from time import time
import torch
from os.path import join, dirname
import copy
from .tf_logger import TFLogger

_log_path = join(dirname(__file__), '../output')  #获取当前文件夹的绝对路径  再跳到上一层的logs文件夹下


# high level wrapper for tf_logger.TFLogger
class Logger():
    def __init__(self, args, update_frequency=10):
        self.current_epoch = 0
        self.max_epochs = args.epochs
        self.last_update = time()
        self.start_time = time()
        self._clean_epoch_stats()
        self.update_f = update_frequency
        folder, logname = self.get_name_from_args(args)   #获得存放的文件夹名称和具体文件名字
        self.folder = folder
        log_path = join(_log_path, str(args.dataset), 'seed'+str(args.seed), 'logs', folder, logname)   #具体文件的绝对路径  这里没有判断文件是否存在，如果不存在生成的操作
        # print(log_path)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        if args.tf_logger:   #开启tensorboard compatible logs   Logger里的属性只是用在tf里的一些参数
            self.tf_logger = TFLogger(log_path)
            # print("Saving to %s" % log_path)
        else:
            self.tf_logger = None
        self.current_iter = 0
        self.res = 0
        self.mod = None
        self.args = args

    def new_epoch(self, learning_rates):   #画学习率变化的
        self.current_epoch += 1
        self.last_update = time()
        self.lrs = learning_rates
        record_new_epoch ="New epoch - lr: %s" % ", ".join([str(lr) for lr in self.lrs])
        self.record_logs(record_new_epoch)      #记录新epoch的日志log
        print(record_new_epoch)
        self._clean_epoch_stats()
        if self.tf_logger:
            for n, v in enumerate(self.lrs):
                self.tf_logger.scalar_summary("aux/lr%d" % n, v, self.current_iter)   #aux是一个标题名  /应该起到了换行的作用

    def log(self, it, iters, losses, samples_right, total_samples):   #log train
        self.current_iter += 1
        loss_string = ", ".join(["%s : %.3f" % (k, v) for k, v in losses.items()])
        for k, v in samples_right.items():
            past = self.epoch_stats.get(k, 0.0)
            self.epoch_stats[k] = past + v
        self.total += total_samples
        acc_string = ", ".join(["%s : %.2f" % (k, 100 * (v / total_samples)) for k, v in samples_right.items()])
        if it % self.update_f == 0:
            record_train_log = "%d/%d of epoch %d/%d %s - acc %s [bs:%d]" % (it, iters, self.current_epoch, self.max_epochs, loss_string,
                                                                acc_string, total_samples)
            self.record_logs(record_train_log)
            print(record_train_log)
            # update tf log
            if self.tf_logger:
                for k, v in losses.items(): self.tf_logger.scalar_summary("train/loss_%s" % k, v, self.current_iter)

    def _clean_epoch_stats(self):  #清除一个epoch的状态
        self.epoch_stats = {}
        self.total = 0

    def log_test(self, phase, accuracies, model, args):   # log test val
        record_test = "Accuracies on %s: " % phase + ", ".join(
            ["%s : %.2f" % (k, v * 100) for k, v in accuracies.items()])
        if phase == "test":
            self.save_model(model, accuracies["class"], args)
            self.record_logs(record_test)
        print(record_test)
        if self.tf_logger:
            for k, v in accuracies.items(): self.tf_logger.scalar_summary("%s/acc_%s" % (phase, k), v, self.current_iter)

    def save_best(self, val_test, best_test):
        print("It took %g" % (time() - self.start_time))
        if self.tf_logger:
            for x in range(10):
                self.tf_logger.scalar_summary("best/from_val_test", val_test, x)
                self.tf_logger.scalar_summary("best/max_test", best_test, x)

    @staticmethod
    def get_name_from_args(args):
        '''文件夹名字'''
        folder_name = "%s_to_%s" % ("-".join(sorted(args.source)), args.target)   #文件夹的名字  源域to目标域
        if args.folder_name:    #默认是test
            folder_name = join(args.folder_name, folder_name)   #在 源域to目标域前加了一层文件夹test

        '''里面文件夹的名字'''
        name = "eps%d_bs%d_lr%g_class%d_jigWeight%g" % (args.epochs, args.batch_size, args.learning_rate, args.n_classes, 0.7)
        # if args.ooo_weight > 0:
        #     name += "_oooW%g" % args.ooo_weight
        if args.train_all:
            name += "_TAll"
        if args.bias_whole_image:
            name += "_bias%g" % args.bias_whole_image
        if args.classify_only_sane:
            name += "_classifyOnlySane"
        if args.TTA:
            name += "_TTA"
        try:
            name += "_entropy%g_jig_tW%g" % (args.entropy_weight, args.target_weight)
        except AttributeError:
            pass
        if args.suffix:
            name += "_%s" % args.suffix
        name += "_%d" % int(time() % 1000)
        return folder_name, name

    def save_model(self, model, acc=None, args=None):

        model_dirs = join('output/', self.args.dataset, 'seed'+str(self.args.seed), 'model', args.target)
        if not os.path.exists(model_dirs):
            os.makedirs(model_dirs)
        if acc > self.res:
            self.res = acc
            self.mod = copy.deepcopy(model)
        if self.current_epoch == args.epochs:
            model_name = str(self.res)[0:8] + "_resnet18.pth"
            model_path = join(model_dirs, model_name)
            print(model_path)
            torch.save(self.mod, model_path)
            # print("模型保存成功")

    def record_default(self, default):
        '''
        保存默认配置文件信息到日志中
        '''
        self.record_logs(record="\nArgs:")
        for k, v in self.args.__dict__.items():
            sen = str(k) + " : " + str(v)
            self.record_logs(sen)
        self.record_logs(record="\nMixup")
        for key, value in default.items():
            senten = str(key) + " : " + str(value)
            self.record_logs(senten)
        self.record_logs(record="\n")

    def record_logs(self, record):
        '''
          保存日志训练文件
        '''
        txt_name = "to_" + str(self.args.target) + '.txt'
        txt_folder = self.folder
        txt_path = join(_log_path, str(self.args.dataset), 'seed'+str(self.args.seed), 'logs', txt_folder, txt_name)
        file = open(txt_path, mode='a+')
        file.write(record + '\n')
