import torch
import torch.nn as nn
import math
import utils.pg_utils as q

def CONV(in_channel, out_channel, kernel_size, stride=1, groups=1,padding = 0,bit = 32):
    if bit == 32:
        return nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, groups = groups)
    else:
        return q.QuantizedConv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, groups = groups,wbits = bit,abits=bit)

class VGG(nn.Module):
    def __init__(self, num_classes=10, bit = 32):
        super(VGG, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            CONV(128, 128, 3, stride=1, padding=1,bit = bit),#in,out,k,s,p
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            CONV(128, 256, 3, stride=1, padding=1, bit = bit),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            CONV(256, 256, 3, stride=1, padding=1, bit = bit),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            CONV(256, 512, 3, stride=1, padding=1, bit = bit),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            CONV(512, 512, 3, stride=1, padding=1, bit = bit),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )
        self.fc = nn.Linear(512*4*4, num_classes)

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out
def vgg(bit):
    return VGG(bit = bit)