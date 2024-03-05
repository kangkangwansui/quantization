import torch
import torch.nn.functional as F


#.....................基于clip可学习的均匀量化方法....................#
class Round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, number, upper, lower):
        in_range = upper - lower  #torch.sub(upper,lower)
        s = in_range / number  #torch.div(in_range , number)

        if s == 0:
            return input
        else:
            return torch.round((input - lower) / s) * s + lower
            #torch.add(torch.mul(torch.round(torch.div((torch.sub(input , lower)) , s)) , s) , lower)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None

class clip_non_quant(torch.autograd.Function):
    @staticmethod
    def forward(ctx,  *args, **kwargs):
        #需要执行的量化操作
        return #xxx

    @staticmethod
    def backward(self, *grad_outputs):
        #反向推理、梯度传播
        return #xxx


class clip_act_fun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        # 需要执行的量化操作
        return  # xxx

    @staticmethod
    def backward(self, *grad_outputs):
        # 反向推理、梯度传播
        return  # xxx