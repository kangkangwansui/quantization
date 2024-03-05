import torch
import torch.nn.functional as F

class Round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, number, upper, lower):
        in_range = upper - lower
        s = in_range / number

        if s == 0:
            return input
        else:
            return ((input - lower) / s).round() * s + lower

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None

class clip_non_quant(torch.autograd.Function):
    @staticmethod
    def forward(ctx,  *args, **kwargs):
        x = args[0]
        a = args[1]
        bit_range = args[2]
        bit_range_pos = bit_range * 0.5
        alpha1 = args[3]
        alpha2 = args[4]

        std = torch.std(x)
        x_min = torch.min(x)
        x_max = torch.max(x)

        if (3 * std < x_max) | (-3 * std > x_min):
            a.data = torch.where(a <= 1, 1.0, a)
            c_p = 3 * std * (1 / a) + (1 - 1 / a) * x_max
            c_n = -3 * std * (1 / a) + (1 - 1 / a) * x_min
        else:
            c_p = x_max
            c_n = x_min

        s = (c_p - c_n) / bit_range
        ctx.save_for_backward(x, std, a, x_max, x_min, c_p, c_n, s)
        x_clip = F.hardtanh(x, min_val=c_n, max_val=c_p)

        x_abs = torch.abs(x_clip)
        x_sign = torch.sign(x_clip)
        mask1 = x_abs < std
        mask2 = (x_abs >= std) & (x_abs < 2 * std)
        mask3 = x_abs >= 2 * std

        number1 = torch.floor(bit_range_pos * alpha1)
        number2 = torch.ceil(bit_range_pos * alpha2)
        number3 = bit_range_pos - number1 - number2

        x_round = torch.zeros_like(x)
        if number1 != 0:
            x_round[mask1] = Round.apply(x_abs[mask1], number1, std, 0)
        if number2 != 0:
            x_round[mask2] = Round.apply(x_abs[mask2], number2, 2 * std, std)
        else:
            x_round[mask2] = torch.max(x_round[mask1])
        if number3 != 0:
            x_round[mask3] = Round.apply(x_abs[mask3], number3, torch.max(abs(x_clip)), 2 * std)
        else:
            x_round[mask3] = torch.max(x_round[mask2])
        return x_round * x_sign

    @staticmethod
    def backward(self, *grad_outputs):
        grad_top = grad_outputs[0]
        x, std, a, x_max, x_min, c_p, c_n, s = self.saved_tensors
        grad_weight = grad_top
        if (3 * std < x_max) | (-3 * std > x_min):
            a_2 = (a ** 2)
            a_grad_n = torch.round((3 * std + x_min) / (a_2 * s)) * s
            a_grad_p = torch.round((x_max - 3 * std) / (a_2 * s)) * s

            internal_flag = ((x <= c_n) | (x >= c_p)).float()
            x_flag_n = torch.where(x < 0, 1.0, 0.0)
            x_flag_p = torch.where(x > 0, 1.0, 0.0)

            a_grad_n = a_grad_n * internal_flag * x_flag_n
            a_grad_p = a_grad_p * internal_flag * x_flag_p
            grad_a = ((a_grad_n + a_grad_p) * grad_top).sum().view((1,))

            return grad_weight, grad_a, None, None, None
        else:
            return grad_weight, None, None, None, None


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