import torch

import torch.nn.functional as F


# ........................基于clip可学习的均匀量化方法量化方法........................#

class clip_weight_fun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        x = args[0]
        a = args[1]
        bit_range = args[2]

        std = torch.std(x)
        x_min = torch.min(x)
        x_max = torch.max(x)

        if (3*std < x_max) | (-3*std > x_min):
            a.data = torch.where(a <= 1, 1.0, a)
            c_p = 3*std * (1 / a) + (1 - 1 / a) * x_max
            c_n = -3*std * (1 / a) + (1 - 1 / a) * x_min
        else:
            c_p = x_max
            c_n = x_min
        x_clip = F.hardtanh(x, min_val=c_n, max_val=c_p)
        s = (c_p - c_n) / bit_range
        ctx.save_for_backward(x, std, a, x_max, x_min, c_p, c_n,s)
        x_round = torch.round(x_clip / s) * s
        return x_round

    @staticmethod
    def backward(self, *grad_outputs):
        grad_top = grad_outputs[0]
        x, std, a, x_max, x_min, c_p, c_n,s = self.saved_tensors
        grad_weight = grad_top
        if (3*std < x_max) | (-3*std > x_min):
            a_2 = (a ** 2)
            a_grad_n = torch.round((3*std + x_min) / (a_2 * s)) * s
            a_grad_p = torch.round((x_max - 3 * std) / (a_2 * s)) * s

            internal_flag = ((x <= c_n) | (x >= c_p)).float()
            x_flag_n = torch.where(x < 0, 1.0, 0.0)
            x_flag_p = torch.where(x > 0, 1.0, 0.0)

            a_grad_n = a_grad_n * internal_flag * x_flag_n
            a_grad_p = a_grad_p * internal_flag * x_flag_p
            grad_a = ((a_grad_n + a_grad_p) * grad_top).sum().view((1,))

            return grad_weight, grad_a, None
        else:
            return grad_weight, None ,None

class clip_act_fun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        x = args[0]
        a = args[1]
        bit_range = args[2]
        device = args[3]

        x_min = torch.min(x)
        x_max = torch.max(x)

        a.data = torch.where(a <= 1, 1.0, a)
        c_p = x_max * (1-0.2/a)
        if x_min >= 0:
            c_n = torch.tensor(0.0).to(device)
        else:
            c_n = x_min * (1-0.2/a)

        x_clip = F.hardtanh(x, min_val=c_n, max_val=c_p)
        s = (c_p - c_n) / bit_range
        ctx.save_for_backward(x, a, x_max, x_min, c_p, c_n, s)
        x_round = torch.round(x_clip / s) * s
        return x_round

    @staticmethod
    def backward(self, *grad_outputs):
        grad_top = grad_outputs[0]
        x, a, x_max, x_min, c_p, c_n, s = self.saved_tensors
        grad_weight = grad_top
        a_2 = (a ** 2)

        if x_min >= 0:
            a_grad_p = 0.2 * (x_max / a_2)
            flag = (x >= c_p).float()
            a_grad = ((a_grad_p * flag) * grad_top).sum().view((1,))
            return grad_weight, a_grad, None, None
        else:
            a_grad_p = 0.2 * (x_max / a_2)
            a_grad_n = 0.2 * (x_min/ a_2)

            internal_flag = ((x <= c_n) | (x >= c_p)).float()
            x_flag_n = torch.where(x < 0, 1.0, 0.0)
            x_flag_p = torch.where(x > 0, 1.0, 0.0)

            a_grad_n = a_grad_n * internal_flag * x_flag_n
            a_grad_p = a_grad_p * internal_flag * x_flag_p
            a_grad = ((a_grad_n + a_grad_p) * grad_top).sum().view((1,))
            return grad_weight, a_grad, None,None




