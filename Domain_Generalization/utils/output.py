import torch
import torchvision
import os.path
from os.path import join

def save_model(model, acc=None, args=None):
    model_dirs = join('output', args.target)
    if not os.path.exists(model_dirs):
        os.makedirs(model_dirs)
    model_name = str(acc) + "resnet18.pth"
    model_path = join(model_dirs, model_name)
    print(model_path)
    torch.save(model, model_path)
    print("模型保存成功")