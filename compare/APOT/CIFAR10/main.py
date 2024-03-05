import argparse
import os
import csv
import time
import shutil
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from tensorboardX import SummaryWriter

import torchvision
import torchvision.transforms as transforms

from models import *

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-m', '--model', metavar='MDELO', default='vgg')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128),only used for train')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('-ct', '--cifar-type', default='10', type=int, metavar='CT', help='10 for cifar10,100 for cifar100 (default: 10)')
parser.add_argument('--init', help='initialize form pre-trained floating point model', type=str, default='')
parser.add_argument('-id', '--device', default='0', type=str, help='gpu device')
parser.add_argument('--bit', default=4, type=int, help='the bit-width of the quantized network')
parser.add_argument('--csv', default='result/csv/', type=str, help='csv保存的路径')
parser.add_argument('--log', default='result/log/', type=str, help='log保存的路径')
parser.add_argument('--dataset', default="cifar10", type=str, help='the bit-width of the quantized network')
parser.add_argument('--quant', default="apot", type=str, help='')
parser.add_argument('--seed', default=2, type=int, help='')



def main(args):
    global best_prec
    best_prec = 0
    use_gpu = torch.cuda.is_available()
    print(args.device)
    print('=> Building model...')
    model=None
    if use_gpu:
        float = True if args.bit == 32 else False
        if args.model == 'res20':
            model = resnet20_cifar(float=float)
        elif args.model == "res18":
            model = resnet18(float=float,quant_mode = args.quant)
        elif args.model == "res32":
            model = resnet32_cifar(float=float)
        elif args.model == "res34":
            model = resnet34(float=float,quant_mode = args.quant)
        elif args.model == "res50":
            model = resnet50(float=float,quant_mode = args.quant)
        elif args.model == 'res56':
            model = resnet56_cifar(float=float)
        elif args.model == 'vgg':
            model = vgg(quant_mode=args.quant)
        else:
            print('Architecture not support!')
            return
        if not float:
            for m in model.modules():
                if isinstance(m, QuantConv2d):
                    m.weight_quant = weight_quantize_fn(w_bit=args.bit)
                    m.act_grid = build_power_value(args.bit)
                    m.act_alq = act_quantization(args.bit, m.act_grid)

        model = model.to(device)
        criterion = nn.CrossEntropyLoss().cuda()
        model_params = []
        for name, params in model.named_parameters():
            if 'act_alpha' in name:
                model_params += [{'params': [params], 'lr': 1e-1, 'weight_decay': 1e-4}]
            elif 'wgt_alpha' in name:
                model_params += [{'params': [params], 'lr': 2e-2, 'weight_decay': 1e-4}]
            else:
                model_params += [{'params': [params]}]
        optimizer = torch.optim.SGD(model_params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        cudnn.benchmark = True
    else:
        print('Cuda is not available!')
        return


    if args.init:
        if os.path.isfile(args.init):
            print("=> loading pre-trained model")
            checkpoint = torch.load(args.init)
            model.load_state_dict(checkpoint['state_dict'],strict=False)
        else:
            print('No pre-trained model found !')
            exit()

    print('=> loading cifar10 data...')
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
    train_dataset = torchvision.datasets.CIFAR10(
        root='/home/zhoukang/quantization_code/Quantization_Code_Pytorch_SSH/dataset/data_cifar10',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    test_dataset = torchvision.datasets.CIFAR10(
        root='/home/zhoukang/quantization_code/Quantization_Code_Pytorch_SSH/dataset/data_cifar10',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

    if args.evaluate:
        validate(testloader, model, criterion)
        model.module.show_params()
        return

    # .........................设置需要记录的参数列表.........................#
    epoch1 = []
    train_loss = []
    train_top1 = []
    train_top5 = []
    val_loss = []
    val_top1 = []
    val_top5 = []
    column_names = ["epoch", "train_loss", "train_top1", "train_top5", "val_loss", "val_top1", "val_top5"]

    if float == True:
        if not os.path.exists(args.log):
            os.makedirs(args.log)
        fdir = str(args.log) + str(args.model) + '_' + str(args.dataset) + "_" + 'float' + '_'  + '_' + str(args.seed)+ '.csv'
        if not os.path.exists(fdir):
            os.makedirs(fdir)

        if not os.path.exists(args.csv):
            os.makedirs(args.csv)
        csv_file = str(args.csv) + str(args.model) + '_' + str(args.dataset) + "_" + 'float' + '_'+ '_' + str(args.seed)+ '.csv'

    else:
        if not os.path.exists(args.log):
            os.makedirs(args.log)
        fdir = str(args.log) +str(args.model)+'_'+ str(args.dataset) + "_" + str(args.quant)+'_'+ str(args.bit)+'bit' + '_'  + '_' + str(args.seed)+ ".csv"
        if not os.path.exists(fdir):
            os.makedirs(fdir)

        if not os.path.exists(args.csv):
            os.makedirs(args.csv)
        csv_file = str(args.csv) + str(args.model)+'_'+ str(args.dataset) + "_" + str(args.quant)+'_'+ str(args.bit)+'bit' + '_' + '_' + str(args.seed)+ ".csv"

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        # model.module.record_weight(writer, epoch)
        epoch1.append(epoch)
        if epoch % 10 == 1:
            print("正在训练APOT的{}模型的{}bit量化方法".format(args.model,args.bit))
            for m in model.modules():
                if isinstance(m, QuantConv2d) :
                    m.show_params()
        # model.module.record_clip(writer, epoch)
        train_p1,train_p5,train_l = train(trainloader, model, criterion, optimizer, epoch)
        train_loss.append(train_l)
        train_top1.append(train_p1)
        train_top5.append((train_p5))

        # evaluate on test set
        val_p1,val_p5,val_l = validate(testloader, model, criterion)
        val_loss.append(val_l)
        val_top1.append(val_p1)
        val_top5.append(val_p5)

        # remember best precision and save checkpoint
        is_best = val_p1 > best_prec
        best_prec = max(val_p1,best_prec)
        print('best acc: {:1f}'.format(best_prec))
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec': best_prec,
            'optimizer': optimizer.state_dict(),
        }, is_best, fdir)
    write_lists_to_csv(csv_file, [epoch1, train_loss, train_top1, train_top5, val_loss, val_top1, val_top5],
                       column_names)
    print("csv文件已保存在：", csv_file)


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


def train(trainloader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        input, target = input.to(device), target.to(device)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1,prec5 = accuracy(output, target,topk = (1,5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if i % 2 == 0:
        #     model.module.show_params()
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                   epoch, i, len(trainloader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
    return top1.avg,top5.avg,losses.avg

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input, target = input.to(device), target.to(device)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1,prec5 = accuracy(output, target,topk=(1,5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1))

    print(' * Prec {top1.avg:.3f}% '.format(top1=top1))

    return top1.avg,top5.avg,losses.avg


def save_checkpoint(state, is_best, fdir):
    filepath = os.path.join(fdir, 'checkpoint.pth')
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(fdir, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    adjust_list = [150, 225]
    if epoch in adjust_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1


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
    return alpha1,alpha2,alpha3

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

if __name__=='__main__':
    global args
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device('cuda' + ':' + args.device if torch.cuda.is_available() else 'cpu')
    config = [["res20", 4],
              ["res20", 3],
              ["res32", 4],
              ["res32", 3],
              ["vgg",4],
              ["vgg",3]]
    for model, bit in config:
        args.model = model
        args.bit = bit
        main(args)
