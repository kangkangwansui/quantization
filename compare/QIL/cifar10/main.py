import os
import csv
import time
import shutil
import datetime
import argparse
import random


import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from torchsummary import summary
from resnet import resnet20,resnet32
from vgg import vgg
from qil import *

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('-m', '--model', metavar='MODEL', default='resnet20',help='models have resnet18、34、50、101、152')
parser.add_argument('--quant', default='non_quant', type=str,help='quantize mode include : clip_quant、non_quant、non_clip_quant、melt_quant、n2uq')
parser.add_argument('-id', '--device', default='0', type=str, help='gpu device')
parser.add_argument('--bit', default=4, type=int, help='the bit-width of the quantized network')
parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default: 128),only used for train')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('-ct', '--cifar-type', default='10', type=int, metavar='CT', help='10 for cifar10,100 for cifar100 (default: 10)')
parser.add_argument('--init', help='initialize form pre-trained floating point model', type=str, default='')
parser.add_argument('--dataset', default="cifar10", type=str, help='the bit-width of the quantized network')
parser.add_argument('--seed', default=7, type=int, help='Generate random weighted data')
parser.add_argument('--csv', default='result/csv/', type=str, help='csv保存的路径')
parser.add_argument('--log', default='result/log/', type=str, help='log保存的路径')
parser.add_argument('--pretrain_model_path', default='', type=str, metavar='PATH', help='')


def main(args):
    global best_prec

    time_begin = get_current_time()

    best_prec = 0
    use_gpu = torch.cuda.is_available()
    print(args.device)
    print('=> Building model...')
    model=None
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    if args.bit == 32:
        pass
    elif args.bit == 5:
        args.pretrain_model_path = args.log + args.model + '_'+args.dataset  + '_best_32_32_model.pt'
    elif args.bit == 4:
        args.pretrain_model_path = args.log + args.model + '_'+args.dataset  + '_best_5_5_model.pt'
    elif args.bit == 3:
        args.pretrain_model_path = args.log + args.model + '_'+args.dataset  + '_best_4_4_model.pt'
    elif args.bit == 2:
        args.pretrain_model_path = args.log + args.model + '_'+args.dataset  + '_best_3_3_model.pt'

    if use_gpu:
        float = True if args.bit == 32 else False
        print("使用{}模型，是否为全精度{}，量化bit为{}，：".format(args.model,float,args.bit))
        if args.model == "resnet20":
            model = resnet20(args.pretrain_model_path, num_classes=10)
        elif args.model == "resnet32":
            model = resnet32(args.pretrain_model_path, num_classes=10)
        elif args.model == "vgg":
            model = vgg(args.pretrain_model_path, num_classes=10)

        else:
            print('Architecture not support!')
            return
        if not float:
            model = set_bit(model, args.bit,args.bit)




        model = model.to(device)
        #summary(model, (3, 224, 224))
        criterion = nn.CrossEntropyLoss().cuda()
        model_params = []
        for name, params in model.named_parameters():
            if 'a_a' in name:
                model_params += [{'params': [params], 'lr': 1e-1, 'weight_decay': 1e-4}]
            elif 'a_w' in name:
                model_params += [{'params': [params], 'lr': 2e-2, 'weight_decay': 1e-4}]

            elif 'act_alpha' in name:
                model_params += [{'params': [params], 'lr': 1e-1, 'weight_decay': 1e-4}]
            elif 'wgt_alpha' in name:
                model_params += [{'params': [params], 'lr': 2e-2, 'weight_decay': 1e-4}]
            else:
                model_params += [{'params': [params]}]
        optimizer = torch.optim.SGD(model_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        cudnn.benchmark = True
    else:
        print('Cuda is not available!')
        return

    current_time = get_current_time()
    if float == True:
        if not os.path.exists(args.log):
            os.makedirs(args.log)
        fdir = str(args.log) + str(args.model) + '_' + str(args.dataset) + "_" + 'float' + '_' + str(current_time) + '_' + str(args.seed)+ '.csv'
        if not os.path.exists(fdir):
            os.makedirs(fdir)

        if not os.path.exists(args.csv):
            os.makedirs(args.csv)
        csv_file = str(args.csv) + str(args.model) + '_' + str(args.dataset) + "_" + 'float' + '_' + str(current_time) + '_' + str(args.seed)+ '.csv'

    else:
        if not os.path.exists(args.log):
            os.makedirs(args.log)
        fdir = str(args.log) +str(args.model)+'_'+ str(args.dataset) + "_" + str(args.quant)+'_'+ str(args.bit)+'bit' + '_' + str(current_time) + '_' + str(args.seed)+ ".csv"
        if not os.path.exists(fdir):
            os.makedirs(fdir)

        if not os.path.exists(args.csv):
            os.makedirs(args.csv)
        csv_file = str(args.csv) + str(args.model)+'_'+ str(args.dataset) + "_" + str(args.quant)+'_'+ str(args.bit)+'bit' + '_' +  str(current_time) + '_' + str(args.seed)+ ".csv"

#.........................设置需要记录的参数列表.........................#
    epoch1 = []
    train_loss = []
    train_top1 = []
    train_top5 = []
    val_loss = []
    val_top1 = []
    val_top5 = []
    column_names = ["epoch","train_loss","train_top1","train_top5","val_loss","val_top1","val_top5"]


    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading pre-trained model")
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            args.start_epoch = checkpoint['epoch']
            epoch1 = checkpoint['epoch']
            train_loss = checkpoint['train_loss']
            train_top1 = checkpoint['train_top1']
            train_top5 = checkpoint['train_top5']
            val_loss = checkpoint['val_loss']
            val_top1 = checkpoint['val_top1']
            val_top5 = checkpoint['val_top5']
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

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        epoch1.append(epoch)


        losses_train, top1_train, top5_train = train(trainloader, model, criterion, optimizer, epoch)
        train_loss.append(losses_train)
        train_top1.append(top1_train)
        train_top5.append((top5_train))

        # evaluate on test set
        losses_val, prec1, prec5 = validate(testloader, model, criterion)
        val_loss.append(losses_val)
        val_top1.append(prec1)
        val_top5.append(prec5)

        # remember best precision and save checkpoint
        is_best = prec1 > best_prec
        best_prec = max(prec1, best_prec)
        if epoch % 10 == 1:
            for m in model.modules():
                if isinstance(m, QConv2d):
                    m.show_params()
        if epoch % 100 == 0:
            write_lists_to_csv(csv_file,[epoch1,train_loss,train_top1,train_top5,val_loss,val_top1,val_top5],column_names)
        print('best acc: {:1f}'.format(best_prec))
        if is_best == True:
            check_point = {'state_dict': model.state_dict(),
                           'optim': optimizer.state_dict(),
                           'EPOCH': epoch,
                           'Acc': best_prec}
            torch.save(check_point, args.log + args.model + '_'+args.dataset  + '_best_{}_{}_model.pt'.format(args.bit,args.bit))
    write_lists_to_csv(csv_file, [epoch1, train_loss, train_top1,train_top5, val_loss, val_top1, val_top5], column_names)
    print("csv文件已保存在：",csv_file)
    time_end = get_current_time()
    total_time = time_compute(time_begin,time_end)
    print("训练代码总共耗时：",total_time)


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
        prec1,prec5 = accuracy(output, target,topk=(1,5))
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
                  'Prec {top1.val:.3f}% ({top1.avg:.3f}%)\t'
                  'Prec {top5.val:.3f}% ({top5.avg:.3f}%)'.format(
                   epoch, i, len(trainloader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1,top5=top5))
    return losses.avg,top1.avg,top5.avg

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
                  'Prec {top1.val:.3f}% ({top1.avg:.3f}%)\t'
                  'Prec {top5.val:.3f}% ({top5.avg:.3f}%)\t'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1,top5=top5))

    print(' * Prec {top1.avg:.3f}% '.format(top1=top1))

    return losses.avg,top1.avg,top5.avg


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
        shutil.copyfile(filename, os.path.join(fdir, 'model_best.pth.tar'))
        shutil.copyfile(filename, os.path.join(fdir, 'model_best.pth'))



def adjust_learning_rate(optimizer, epoch):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    adjust_list = [150, 200,250,275]
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

def set_bit(model,w_bit,a_bit):
    for m in model.modules():
        if isinstance(m, QConv2d):
            m.bit = w_bit
        if isinstance(m, QActivation):
            m.bit = a_bit
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

if __name__=='__main__':
    args = parser.parse_args()
    args.device = '2'
    device = torch.device('cuda' + ':' + args.device if torch.cuda.is_available() else 'cpu')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    config = [["resnet32", 5],
              ["resnet32", 4],
              ["resnet32", 3],
              ["resnet32", 2],
              ]
    for model, bit in config:
        num = 0
        while (1):
            args.model = model
            args.bit = bit
            nan = main(args)
            if nan == False:
                break
            else:
                random_integer = random.randint(1, 100)
                args.seed = random_integer
                num += 1

            if num == 7:
                break