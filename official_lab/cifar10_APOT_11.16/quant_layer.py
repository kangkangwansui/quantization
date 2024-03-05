import torch
import torch.nn as nn
import torch.nn.functional as F

from method import clip_non,non,clip_sy

from args import args

device = torch.device('cuda' + ':' + args.device if torch.cuda.is_available() else 'cpu')
class CONV(nn.Conv2d):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(CONV,self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,bias)
        self.bit = 8
        self.bit_range = 2 ** self.bit
        self.quant_mode = "conv"
        self.a_w = nn.Parameter(data=torch.tensor(5.0))
        self.a_a = nn.Parameter(data=torch.tensor(5.0))
        self.beta = nn.Parameter(data=torch.tensor(0.0))
        self.alpha1 = nn.Parameter(data=torch.tensor(0.65))
        self.alpha2 = nn.Parameter(data=torch.tensor(0.3))

    def forward(self,x):
        if self.quant_mode == "conv":
            x_q = x
            w_q = self.weight
        elif self.quant_mode == "clip_asy":
            x_q = clip_asy.clip_act_fun.apply(x,self.a_a,self.bit_range)
            w_q = clip_asy.clip_weight_fun.apply(self.weight,self.a_w,self.beta,self.bit_range)
        elif self.quant_mode == "clip_melt":
            x_q = clip_melt.clip_act_fun.apply(x,self.bit_range)
            w_q = clip_melt.clip_weight_fun.apply(self.weight,self.bit_range)
        elif self.quant_mode == "clip_sy":
            x_q = clip_sy.clip_act_fun.apply(x,self.a_a,self.bit_range,device)
            w_q = clip_sy.clip_weight_fun.apply(self.weight,self.a_w,self.bit_range)
        elif self.quant_mode == "clip_non":
            x_q = clip_non.clip_act_fun.apply(x,self.a_a,self.bit_range,device)
            w_q = clip_non.clip_non_quant.apply(self.weight,self.a_w,self.bit_range,self.alpha1,self.alpha2)
        elif self.quant_mode == "non":
            x_q = non.clip_act_fun.apply(x,self.a_a,self.bit_range,device)
            w_q = non.non_quant.apply(self.weight,self.bit_range,self.alpha1,self.alpha2)
        else:
            print("请输入正确的量化方法")
        return F.conv2d(x_q, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups)

class my_conv(nn.Conv2d):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(my_conv,self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,bias)
        self.bit = 8
        self.bit_range = 2 ** self.bit
        self.quant_mode = "conv"
        self.a_w = nn.Parameter(data=torch.tensor(5.0))
        self.a_a = nn.Parameter(data=torch.tensor(5.0))
        self.beta = nn.Parameter(data=torch.tensor(0.0))
        self.alpha1 = torch.tensor(0.65).to(device)
        self.alpha2 = torch.tensor(0.3).to(device)


    def froward(self,x):
        return F.conv2d(x,self.weight,self.bias,self.stride, self.padding, self.dilation, self.groups)


def conv3x3(in_planes, out_planes, stride=1,padding = 1):
    " 3x3 convolution with padding "
    return my_conv(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    return my_conv(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def Quantconv3x3(in_planes, out_planes, stride=1,padding = 1):
    " 3x3 quantized convolution with padding "
    return CONV(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False)

def Quantconv1x1(in_planes, out_planes, stride=1,padding = 1):
    " 3x3 quantized convolution with padding "
    return CONV(in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, bias=False)

def Quantconv(in_planes, out_planes, kernel_size = 3,stride=1,padding = 1,groups = 1,bias = False):
    return CONV(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias,groups=groups)