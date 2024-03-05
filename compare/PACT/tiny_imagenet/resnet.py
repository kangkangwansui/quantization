"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn
from quant_layer import QuantizedConv2d

def CONV(in_channel, out_channel, kernel_size, stride=1, groups=1,padding = 0,quant_mode="conv"):
    assert quant_mode in ["conv","pact"]
    if quant_mode == "pact":
        return QuantizedConv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, groups = groups)
    else:
        return nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, groups = groups)

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1,quant_mode = "conv"):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            CONV(in_channels, out_channels, kernel_size=3, stride=stride, padding=1,quant_mode=quant_mode),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            CONV(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1,quant_mode=quant_mode),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                CONV(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride,quant_mode=quant_mode),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=200,quant_mode = "conv"):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1, quant_mode)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2, quant_mode)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2, quant_mode)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2, quant_mode)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride, quant_mode):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride,quant_mode=quant_mode))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

class ResNet_cifar100(nn.Module):

    def __init__(self, block, num_block, num_classes=200,quant_mode = "conv"):
        super().__init__()

        self.in_channels = 16

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 16, num_block[0], 1,quant_mode)
        self.conv3_x = self._make_layer(block, 32, num_block[1], 2,quant_mode)
        self.conv4_x = self._make_layer(block, 64, num_block[2], 2,quant_mode)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride,quant_mode):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride,quant_mode = quant_mode))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output
def resnet20(quant_mode):
    model = ResNet_cifar100(BasicBlock, [3, 3, 3],quant_mode=quant_mode)
    return model


def resnet32(quant_mode):
    model = ResNet_cifar100(BasicBlock, [5, 5, 5],quant_mode=quant_mode)
    return model


def resnet44(quant_mode):
    model = ResNet_cifar100(BasicBlock, [7, 7, 7],quant_mode=quant_mode)
    return model


def resnet56(quant_mode):
    model = ResNet_cifar100(BasicBlock, [9, 9, 9],quant_mode=quant_mode)
    return model
def resnet18(quant_mode):
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2],quant_mode = quant_mode)

def resnet34(quant_mode):
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3],quant_mode = quant_mode)

def resnet50(quant_mode):
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3],quant_mode = quant_mode)

def resnet101(quant_mode):
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3],quant_mode = quant_mode)

def resnet152(quant_mode):
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3],quant_mode = quant_mode)



