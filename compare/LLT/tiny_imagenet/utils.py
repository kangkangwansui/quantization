""" helper function

author baiyu
"""
import os
import sys
import re
import csv
import datetime

import numpy

import torch
from quant_layer import QuantConv2d
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

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

def get_network(args):
    """ return given network
    """

    if args.net == 'resnet18':
        from resnet import resnet18
        net = resnet18(args.quant)
        print("使用{}模型训练，量化bit为{}".format(args.net,args.bit))
    elif args.net == 'resnet20':
        from resnet import resnet20
        net = resnet20(args.quant)
        print("使用{}模型训练，量化bit为{}".format(args.net,args.bit))
    elif args.net == 'resnet32':
        from resnet import resnet32
        net = resnet32(args.quant)
        print("使用{}模型训练，量化bit为{}".format(args.net,args.bit))
    elif args.net == 'resnet34':
        from resnet import resnet34
        net = resnet34(args.quant)
        print("使用{}模型训练，量化bit为{}".format(args.net,args.bit))
    elif args.net == 'resnet50':
        from resnet import resnet50
        net = resnet50(args.quant)
        print("使用{}模型训练，量化bit为{}".format(args.net, args.bit))
    elif args.net == 'resnet101':
        from resnet import resnet101
        net = resnet101(args.quant)
        print("使用{}模型训练，量化bit为{}".format(args.net, args.bit))
    elif args.net == 'resnet152':
        from resnet import resnet152
        net = resnet152(args.quant)
        print("使用{}模型训练，量化bit为{}".format(args.net, args.bit))

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.device: #use_gpu
        device = torch.device('cuda' + ':' + args.device if torch.cuda.is_available() else 'cpu')
        net = net.to(device)

    return net


def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    cifar100_training = torchvision.datasets.CIFAR100(root='/home/zhoukang/quantization_code/dataset/data_Cifar100', train=True, download=True, transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader

def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar100_test = torchvision.datasets.CIFAR100(root='/home/zhoukang/quantization_code/dataset/data_Cifar100', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader

def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std

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


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]

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

def set_bit(model,bit):
    for m in model.modules():
        if isinstance(m, QuantConv2d):
            m.w_bits = bit
            m.a_bits = bit
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