import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from resnet import *
from vgg import *
import datetime
import csv


parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-a', '--arch', metavar='ARCH', default='vgg')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128),only used for train')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('-ct', '--cifar-type', default='10', type=int, metavar='CT', help='10 for cifar10,100 for cifar100 (default: 10)')
parser.add_argument('-id', '--device', default='0', type=str, help='gpu device')
parser.add_argument('--csv', default='result/csv/', type=str, help='gpu device')
parser.add_argument('--w_bits', default=4, type=int, help='the bit-width of the quantized network')
parser.add_argument('--a_bits', default=4, type=int, help='the bit-width of the quantized network')



def main(args):
    time_begin = get_current_time()
    global best_prec

    best_prec = 0

    use_gpu = torch.cuda.is_available()
    print('=> Building model...')

    if use_gpu:
        if args.w_bits != 32 or args.a_bits != 32:
            print("使用{}模型，使用LLQ量化方法，量化bit为{}，：".format(args.arch, args.w_bits))
        if args.arch == 'resnet20':
            model = resnet20_cifar(w_bits=args.w_bits, a_bits=args.a_bits)
        elif args.arch == 'resnet32':
            model = resnet32_cifar(w_bits=args.w_bits, a_bits=args.a_bits)
        elif args.arch == 'vgg':
            model = VGG(w_bits=args.w_bits, a_bits=args.a_bits)
        else:
            print('Architecture not support!')
            return

        model = nn.DataParallel(model.cuda(), list(map(int, args.device.split(','))))
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

        if args.resume:
            # load pre_trained full-precision model
            print("=> loading pre-trained model")
            if args.arch == 'resnet20':
                ckpt = torch.load('result/' + args.arch + '_32bit_best.pth')
            if args.arch == 'vgg':
                ckpt = torch.load('result/' + args.arch + '_32bit_best.pth')
            model.load_state_dict(ckpt, strict=False)

            cudnn.benchmark = True
        else:
            pass
    else:
        print('Cuda is not available!')
        return

    if not os.path.exists('result/log'):
        os.makedirs('result/log')
    fdir = 'result/log/'+str(args.arch) + '_w' + str(args.w_bits) + '_a' + str(args.a_bits)
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    epoches = []
    val_loss = []
    val_top1 = []
    val_top5 = []
    column_names = ["epoch","val_loss", "val_top1", "val_top5"]
    current_time = get_current_time()
    csv_file = str(args.csv) + str(args.arch)+'_'+ str(args.cifar_type) + "_" + "LLQ"+'_'+ str(args.w_bits)+'bit' + '_' +  str(current_time) + ".csv"

    print('=> loading cifar10 data...')
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
    train_dataset = torchvision.datasets.CIFAR10(
        root='/home/zhoukang/quantization_code/dataset/data_cifar10/',
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
        root='/home/zhoukang/quantization_code/dataset/data_cifar10/',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

    for epoch in range(args.start_epoch, args.epochs):
        epoches.append(epoch)
        # update learning rate
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(trainloader, model, criterion, optimizer, epoch)

        # evaluate on test set
        prec1,prec5,val_losses = validate(testloader, model, criterion)
        val_loss.append(val_losses)
        val_top1.append(prec1)
        val_top5.append(prec5)


        if epoch % 100 == 0:
            write_lists_to_csv(csv_file,[epoches,val_loss,val_top1,val_top5],column_names)
        if epoch % 10 == 0:
            print("正在训练LLQ量化方法的{}模型的{}bit权重，{}bit激活量化".format(args.arch,args.w_bits,args.a_bits))

        is_best = prec1 > best_prec
        best_prec = max(prec1, best_prec)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            },is_best,fdir)

        print('best acc: {:1f}'.format(best_prec))
    write_lists_to_csv(csv_file, [epoches,val_loss, val_top1, val_top5], column_names)
    print("csv文件已保存在：", csv_file)
    time_end = get_current_time()
    total_time = time_compute(time_begin, time_end)
    print("训练代码总共耗时：", total_time)


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

    model.train()
    end = time.time()

    for i, (input, target) in enumerate(trainloader):
        # update tau
        tau = max(1 * (0.001 ** ((epoch * 392 + i) / 40 / 392)), 0.001)
        for m in model.modules():
            if hasattr(m, '_update_tau'):
                m.tau = tau

        # measure data loading time
        data_time.update(time.time() - end)

        input, target = input.cuda(), target.cuda()

        # compute output
        output = model(input)

        # losses
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec = accuracy(output, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                   epoch, i, len(trainloader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    for m in model.modules():
        if hasattr(m, '_quantization'):
            m._quantization()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input, target = input.cuda(), target.cuda()

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


def adjust_learning_rate(optimizer, epoch):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    if args.arch == 'vgg':
        adjust_list = [100, 175,225,250]
    elif args.arch == 'resnet20':
        adjust_list = [100, 175,225,250]
    elif args.arch == 'resnet32':
        adjust_list = [100, 175,225,250]
    else:
        adjust_list = [150, 200, 250]
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

def get_current_time():
    current_time = datetime.datetime.now()
    return current_time.strftime("%Y-%m-%d-%H:%M:%S")

import shutil
def save_checkpoint(state, is_best, fdir):
    if os.path.exists('checkpoint.pth'):
        filename = fdir + 'checkpoint.pth'
        checkpoint = torch.load(filename, map_location=device)
        if state['best_acc1'] > checkpoint['best_acc1']:
            torch.save(state, filename)
        else:
            pass
    else:
        filename = os.path.join(fdir, 'checkpoint.pth')
        torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(fdir, 'model_best.pth'))

if __name__=='__main__':
    global args
    args = parser.parse_args()
    device = torch.device('cuda' + ':' + args.device if torch.cuda.is_available() else 'cpu')
    config = [["resnet32", 4],
              ["resnet32", 3],
              ["resnet32", 2]
              ]  # model,bit
    for model, bit in config:
        args.arch = model
        args.w_bits = bit
        args.a_bits = bit
        main(args)