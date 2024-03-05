""" helper function

author baiyu
"""
import os
import sys
import csv
import time
import datetime
import shutil
import random

import numpy as np

import torch
from torch.optim.lr_scheduler import _LRScheduler
from args import args
from quant_layer import CONV

device = torch.device('cuda' + ':' + args.device if torch.cuda.is_available() else 'cpu')

class ProgressMeter(object):   #提供一个可视化的进度条
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(epoch, net, args, trainloader, optimizer,criterion,warmup_scheduler):
    losses = AverageMeter()

    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(trainloader):

        if args.device:
            labels = labels.to(device)
            images = images.to(device)

        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), images.size(0))

        if batch_index % 200 == 0:
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                loss.item(),
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * args.batch_size + len(images),
                total_samples=len(trainloader.dataset)
            ))

        if epoch <= args.warm:
            warmup_scheduler.step()

        if batch_index % 100 == 0:
            if args.quant == 'clip_non':
                for m in net.modules():
                    if isinstance(m, CONV):
                        m.alpha1.data, m.alpha2.data = adjust_alpha(m.weight, m.weight.grad)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
    return losses.avg

@torch.no_grad()
def eval_training(net, args, testloader, criterion,epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    start = time.time()
    net.eval()

    for (images, labels) in testloader:

        if args.device:
            images = images.to(device)
            labels = labels.to(device)

        outputs = net(images)
        loss = criterion(outputs, labels)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs, labels, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(prec1.item(), images.size(0))
        top5.update(prec5.item(), images.size(0))

    finish = time.time()
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Top1 accuracy: {:.4f},Top5 accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        losses.avg,
        top1.avg,
        top5.avg,
        finish - start
    ))
    print()

    return top1.avg,top5.avg,losses.avg

def save_checkpoint(state, is_best, fdir):
    filepath = os.path.join(fdir, 'checkpoint.pth')
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(fdir, 'model_best.pth'))

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def adjust_alpha(weight,weight_grad):
    std = torch.std(weight)
    weight_abs = torch.abs(weight)
    mask1 = weight_abs < std
    mask2 = (weight_abs > std) & (weight_abs <= 2 * std)
    alpha1 = abs(weight_grad[mask1]).sum()
    alpha2 = abs(weight_grad[mask2]).sum()
    alpha3 = abs(weight_grad).sum()
    alpha1 = alpha1 / alpha3
    alpha2 = alpha2 / alpha3
    return alpha1,alpha2

def get_current_time():
    current_time = datetime.datetime.now()
    return current_time.strftime("%Y-%m-%d-%H:%M:%S")

def set_bit(model,args):
    for m in model.modules():
        if isinstance(m, CONV) :
            m.bit = args.bit
            m.bit_range = 2 ** args.bit
            m.quant_mode = args.quant
    return model

def time_compute(time_early,time_later):
    # 将字符串解析为datetime对象
    time1 = datetime.datetime.strptime(time_early, "%Y-%m-%d-%H:%M:%S")
    time2 = datetime.datetime.strptime(time_later, "%Y-%m-%d-%H:%M:%S")

    # 计算时间差
    time_difference = time2 - time1

    # 提取天、小时和分钟
    days = time_difference.days
    seconds = time_difference.seconds
    hours, remainder = divmod(seconds, 3600)
    minutes, _ = divmod(remainder, 60)

    # 格式化输出
    formatted_duration = f"{days}天 {hours}小时 {minutes}分钟"
    return formatted_duration

def get_network(args):
    """ return given network
    """

    if args.model == 'resnet18':
        from resnet import resnet18
        net = resnet18()
        print("使用{}模型训练，量化bit为{}".format(args.model,args.bit))
    elif args.model == 'resnet34':
        from resnet import resnet34
        net = resnet34()
        print("使用{}模型训练，量化bit为{}".format(args.model,args.bit))
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.device: #use_gpu
        device = torch.device('cuda' + ':' + args.device if torch.cuda.is_available() else 'cpu')
        net = net.to(device)

    return net

def get_optimizer(model,args):
    model_params1 = []
    for name, params in model.named_parameters():
        if 'a_w' in name:
            model_params1 += [{'params': [params], 'lr': 1e-1, 'weight_decay': 1e-4}]
            params.data = torch.tensor(5.0).to(device)
        elif 'beta' in name:
            model_params1 += [{'params': [params], 'lr': 1e-1, 'weight_decay': 1e-4}]
        elif 'a_a' in name:
            model_params1 += [{'params': [params], 'lr': 1e-1, 'weight_decay': 1e-4}]
            params.data = torch.tensor(5.0).to(device)
        else:
            model_params1 += [{'params': [params]}]
    optimizer1 = torch.optim.SGD(model_params1, lr=0.1, momentum=args.momentum, weight_decay=args.weight_decay)
    return optimizer1

def load_model(model,args,optimizer1):
    load_file = args.load + 'model_best.pth'
    if not os.path.exists(load_file):
        print("路径{}下没有文件model_best.pth".format(args.load))
    else:
        print("加载全精度模型=============>>>>>")
        checkpoint = torch.load(load_file)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

        print("加载完毕=============>>>>>")
    return model,optimizer1

def save_config(args):
    current_time = get_current_time()
    log_file = args.log + str(args.model) + '_' + str(args.dataset)
    csv_file = args.csv + str(args.model) + '_' + str(args.dataset)
    csv_a_file = args.csv_a + str(args.model) + '_' + str(args.dataset)
    if not os.path.exists(log_file):
        os.makedirs(log_file)
    if not os.path.exists(csv_file):
        os.makedirs(csv_file)
    if not os.path.exists(csv_a_file):
        os.makedirs(csv_a_file)

    log_dir = log_file + '/' + str(args.quant)+'_'+ str(args.bit)+'bit' + '_' + str(args.seed)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    csv_dir = csv_file + '/' + str(args.quant)+'_'+ str(args.bit)+'bit' + '_' + str(args.seed) + '_'+str (current_time) +'.csv'
    csv_a_dir = csv_a_file + '/' + str(args.quant) + '_' + str(args.bit) + 'bit' + '_' + str(args.seed) + '_'+str (current_time) + '.csv'

    return log_dir,csv_dir,csv_a_dir

def write_list_and_matrix_to_csv(list1, matrix, filename):
    """
    将列表 list1 写入 CSV 文件的第一行，然后将矩阵 matrix 写入 CSV 文件的后续行。

    参数：
    - list1: 包含数据的列表
    - matrix: 包含数据的 NumPy 矩阵
    - filename: 要写入的 CSV 文件名
    """
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(list1)
        for row in matrix:
            writer.writerow(row)

def make_alpha_matrix(model,epoch,dir):
    if epoch == 0:
        global column_names
        column_names = []
        column_names.append('epoch')

    column_names2_value = []
    column_names2_value.append(epoch)
    for name, para in model.named_parameters():
        if 'a_a' in name:
            if epoch == 0:
                column_names.append(name)
            column_names2_value.append(para.item())
        elif 'a_w' in name:
            if epoch == 0:
                column_names.append(name)
            column_names2_value.append(para.item())
    if epoch == 0:
        global matrix
        matrix = np.empty((0, len(column_names2_value)))
    matrix = np.vstack([matrix, column_names2_value])

    if (epoch % 100 == 0) & (epoch != 0):
        write_list_and_matrix_to_csv(column_names,matrix,dir)
    if epoch == (args.epochs - 1):
        write_list_and_matrix_to_csv(column_names, matrix, dir)


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def write_lists_to_csv(filename, lists, column_names):
    if len(lists) != len(column_names):
        print("Error: Number of lists should match the number of column names.")
        return

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        # 写入列表名称作为 CSV 文件的第一行
        writer.writerow(column_names)

        # 获取最大列表长度，以防止不等长的列表导致写入错误
        max_length = max(len(lst) for lst in lists)

        # 逐行写入数据
        for i in range(max_length):
            row = [lst[i] if i < len(lst) else "" for lst in lists]
            writer.writerow(row)

def make_acc_csv(train_loss,val_loss,val_top1,val_top5,epoch,best_acc1,best_acc5,dir):
    column_names = ["epoch", "train_loss", "val_loss", "val_top1", "val_top5"]
    if epoch == 0:
        global epoches,train_losses,val_losses,val_top1s,val_top5s

        epoches = []
        train_losses = []
        val_losses = []
        val_top1s = []
        val_top5s = []

        epoches.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_top1s.append(val_top1)
        val_top5s.append(val_top5)

    else:
        epoches.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_top1s.append(val_top1)
        val_top5s.append(val_top5)

    if (epoch % 10 == 0) & (epoch != 0):
        write_lists_to_csv(dir,[epoches,train_losses,val_losses,val_top1s,val_top5s],column_names)

    if epoch == (args.epochs - 1) :
        val_top1s.append(best_acc1)
        val_top5s.append(best_acc5)
        write_lists_to_csv(dir, [epoches,train_losses,val_losses,val_top1s,val_top5s], column_names)

def show_config(args,epoch):
    if epoch % 10 == 0:
        print('正在利用{}模型，在数据集{}上训练{}bit的{}方法'.format(args.model,args.dataset,args.bit,args.quant))


