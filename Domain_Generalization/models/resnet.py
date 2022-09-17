from torch import nn
from torch.utils import model_zoo
from torchvision.models.resnet import BasicBlock, model_urls, Bottleneck
import torch
from torch import nn as nn
from torch.autograd import Variable
import numpy.random as npr
import numpy as np
import torch.nn.functional as F
import random
import math

'''
resnet 认为深层的网络可以提取出更加丰富的语义信息。随着网络的加深一般会让分辨率降低而让通道数增加
resnet18 在第2,3,4,5个stage中，在每个stage中使用的基本模块数目是[2,2,2,2]
'''
class ResNet(nn.Module):
    def __init__(self, block, layers, jigsaw_classes=1000, classes=100):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])        #对应resnet中第2,3,4,5的stage
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.jigsaw_classifier = nn.Linear(512 * block.expansion, jigsaw_classes)
        self.class_classifier = nn.Linear(512 * block.expansion, classes)
        #self.domain_classifier = nn.Linear(512 * block.expansion, domains)
        self.pecent = 1/3    #在一个bc中，选择1/3的样本进行RSC

        for m in self.modules():   #初始化权重
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, stride=1):
        '''

        :param block:  基本模块选择谁（基本模块包括BasicBlock 和 Bottleneck）
        :param planes:  这是每个stage中，与每个block的输出通道相关的参数
        :param blocks:  2
        :param stride:
        :return:
        '''
        downsample = None      #定义了一个下采样模块
        '''
        只要stride>1 或者 输入和输出通道数目不一样，就可以断定残差模块产生的feature map相比于原来的分辨率降低了，此时需要进行下采样
        BasicBlock(或Bottleneck类）中的属性expansion，用于指定下一个BasicBlock的输入通道是多少
        
        '''
        if stride != 1 or self.inplanes != planes * block.expansion:   #前面没有实例化BasicBlock,所以不能使用实例属性，而是直接使用了类属性expansion
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))   #这才是生成了BasicBlock的实例  #self.inplanes等于上一个stage的输出通道数
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):   #resnet18中blocks=2
            layers.append(block(self.inplanes, planes))   #每一个block的输出通道数

        return nn.Sequential(*layers)

    def is_patch_based(self):
        return False

    def forward(self, x, gt=None, flag=None, epoch=None, return_features=False, forward_feature=False):
        if forward_feature:   #CuMix
            return self.class_classifier(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # feature = x

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if return_features:
            # feature = self.avgpool(feature)
            # feature = feature.view(feature.size(0), -1)
            # feature = x
            return self.class_classifier(x), x
        return self.class_classifier(x)


def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)  #并没有把BasicBlock类实例化
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model

def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model
