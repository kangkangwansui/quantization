import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.quant_layer import *


class BasicBlock(nn.Module):
    expansion=1
    def __init__(self, inplanes, planes, quant_mode = "non_quant",stride=1, downsample=None, float=False):
        super(BasicBlock, self).__init__()
        self.quant_mode = quant_mode
        if float:
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.conv2 = conv3x3(planes, planes)
        else:
            assert self.quant_mode in ["non_quant", "non_clip_quant", "clip_quant","melt_quant","apot"], "请输入正确的量化方法"
            if self.quant_mode == "apot":
                self.conv1 = Quantconv3x3(inplanes, planes, stride)
                self.conv2 = Quantconv3x3(planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion=4
    def __init__(self, inplanes, planes,quant_mode = "non_quant", stride=1, downsample=None,float=False):
        super(Bottleneck, self).__init__()
        self.quant_mode = quant_mode
        if float:
            self.conv1 = conv1x1(inplanes, planes)
            self.conv2 = conv3x3(planes, planes,stride = stride)
            self.conv3 = conv1x1(planes, planes*4)
        else:
            assert self.quant_mode in ["non_quant", "non_clip_quant", "clip_quant", "melt_quant","apot"], "请输入正确的量化方法"
            if self.quant_mode == "apot":
                self.conv1 = Quantconv1x1(inplanes, planes)
                self.conv2 = Quantconv3x3(planes, planes, stride=stride)
                self.conv3 = Quantconv1x1(planes, planes * 4)

        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers,quant_mode, num_classes=10, float=False):
        super(ResNet, self).__init__()
        self.float = float
        self.quant_mode = quant_mode
        assert self.quant_mode in ["non_quant", "non_clip_quant", "clip_quant","melt_quant","apot"], "请输入正确的量化方法"
        self.inplanes = 64
        self.conv1 = first_conv(3, self.inplanes, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], float=float)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, float=float)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, float=float)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, float=float)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = last_fc(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, float=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.quant_mode == "apot":
                Qconv = QuantConv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)
            else:
                pass
            downsample = nn.Sequential(
                Qconv if float is False else nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1,
                                                 stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes,self.quant_mode, stride, downsample, float=float))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,self.quant_mode, float=float))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def show_params(self):
        for m in self.modules():
            if isinstance(m, QuantConv2d):
                m.show_params()

def resnet18(**kwargs):
    model = ResNet(BasicBlock, [2,2,2,2], **kwargs)
    return model

def resnet34(**kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def resnet50(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

def resnet101(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model

def resnet152(**kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model

if __name__ == '__main__':
    pass
    # net = resnet18(num_classes = 10)
    # y = net(torch.randn(1, 3, 64, 64))
    # print(net)
    # print(y.size())


