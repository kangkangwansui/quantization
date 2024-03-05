import torch
import torch.nn as nn
import math
import os
from qil import *
def CONV(in_channel, out_channel, kernel_size, stride=1, groups=1,padding = 0,bit = 32):
    if bit == 32:
        return nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, groups = groups)
    else:
        return QConv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, groups = groups,bit = bit)

class VGG(nn.Module):
    def __init__(self, num_classes=10, w_bits=4, a_bits=4):
        super(VGG, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            QActivation(a_bits),
            CONV(128, 128, 3, 1, 1, bit=w_bits),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            QActivation(a_bits),
            CONV(128, 256, 3, 1, 1, bit=w_bits),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            QActivation(a_bits),
            CONV(256, 256, 3, 1, 1, bit=w_bits),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            QActivation(a_bits),
            CONV(256, 512, 3, 1, 1, bit=w_bits),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            QActivation(a_bits),
            CONV(512, 512, 3, 1, 1, bit=w_bits),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )
        self.fc = nn.Linear(512*4*4, num_classes)

    def forward(self, x):
        fea = self.body(x)
        fea = fea.view(fea.size(0), -1)
        out = self.fc(fea)

        return out

def vgg(pretrained_path, num_classes):
    vgg = VGG(num_classes=num_classes,w_bits=32,a_bits=32)

    if pretrained_path is not None:
        if not os.path.exists(pretrained_path):
            print(f'CIFAR10 FP32 不存在')
            return vgg

        checkpoint = torch.load(f'{pretrained_path}')
        vgg.load_state_dict(checkpoint['state_dict'], strict=True)
    return vgg