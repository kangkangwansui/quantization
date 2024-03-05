import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, ImageFolder
from torchvision.transforms import transforms as T
from resnet import resnet20, resnet18,resnet32
from vgg import vgg

import torch.optim as optim
from tqdm import tqdm
from utils import Hook, interval_param_list, weight_param_list
from qil import Transformer
from logger import get_logger
import argparse
import os
import random
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel, DataParallel
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
import time
import csv
import datetime
from qil import QConv2d,QActivation


""" Logger 등록 """
logging = get_logger('./log', log_config='./logging.json')
model_logger = logging.getLogger('model-logger')
param_logger = logging.getLogger('param-logger')

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
""" Data Setting """
def custom_transfroms_cifar10(train=True):

    if train:
        return T.Compose([ T.RandomCrop(32, padding=4),
                           T.RandomHorizontalFlip(),
                           T.ToTensor(),
                           T.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
                           ])
    else:
        return T.Compose([ T.ToTensor(),
                           T.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
                           ])


def custom_transfroms_imagenet(train=True):

    if train:
        return T.Compose([ T.RandomResizedCrop(224),
                           T.RandomHorizontalFlip(),
                           T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                           ])
    else:
        return T.Compose([ T.Resize(256),
                           T.CenterCrop(224),
                           T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                           ])


def train(model, criterion, optimizer, lr_scheduler, train_loader, EPOCH, args):

    model.train()
    train_loss = .0
    with tqdm(train_loader) as tbar:
        for i, data in enumerate(tbar):
            optimizer.zero_grad()
            imgs, targets = data[0],data[1]

            imgs, targets = imgs.to(device), targets.to(device)

            output = model(imgs)
            iter_loss = criterion(output, targets)
            train_loss += iter_loss

            iter_loss.backward()
            optimizer.step()

            tbar.set_description(
                f'EPOCH: {EPOCH} | total_train_loss: {train_loss / (i + 1):.4f} | batch_loss: {iter_loss:.4f}'
            )
        lr_scheduler.step()


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
            input, target = input.to(device),target.to(device)

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

def initialize_param(model, train_set, train_sampler, interval_lr, weight_lr):

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = optim.SGD([{'params': interval_param_list(model), 'lr': interval_lr},
                           {'params': weight_param_list(model), 'lr': weight_lr}], momentum=0.9, weight_decay=1e-4)
    """ Hook """
    handle_hooks = []
    for module_name, module in model.named_modules():
        if isinstance(module, Transformer):
            h = Hook(module, module_name)
            handle_hooks.append(h)

    """ Batch_size만큼 Iteration 1회를 통해서 c_x, d_x 초기값 구하기 """

    model.to(device)
    model.train()
    tmp_dataloader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.workers,
                                shuffle=(train_sampler is None),
                                sampler=train_sampler, pin_memory=True)

    """ c_delta, d_delta 초기값 설정 """
    data = next(iter(tmp_dataloader))
    optimizer.zero_grad()
    imgs, targets = data
    imgs = imgs.to(device)
    targets = targets.to(device)

    output = model(imgs)
    iter_loss = criterion(output, targets)
    iter_loss.backward()


    """ 등록된 hook 제거(메모리관리) """
    for handle_hook in handle_hooks:
        handle_hook.handle.remove()

    print('c_delta, d_delta 초기값 설정 완료'    )


    criterion = None
    optimizer = None
    lr_scheduler = None

def set_bit(model,w_bit,a_bit):
    for m in model.modules():
        if isinstance(m, QConv2d):
            m.bit = w_bit
        if isinstance(m, QActivation):
            m.bit = a_bit
    return model

def main(args):

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    global best_prec
    best_prec = 0

    """ CONFIG """
    w_bit, a_bit = args.w_bit,args.a_bit

    if w_bit == 32:
        pass
    elif w_bit == 5:
        args.pretrain_model_path = args.log + args.model + '_'+args.data  + '_best_32_32_model.pt'
    elif w_bit == 4:
        args.pretrain_model_path = args.log + args.model + '_'+args.data  + '_best_5_5_model.pt'
    elif w_bit == 3:
        args.pretrain_model_path = args.log + args.model + '_'+args.data  + '_best_4_4_model.pt'
    elif w_bit == 2:
        args.pretrain_model_path = args.log + args.model + '_'+args.data  + '_best_3_3_model.pt'


    if os.path.exists(args.pretrain_model_path):
        print(f'pretrain_model_path: {args.pretrain_model_path}')


    if args.device is not None:
        print(f'Use GPU: {args.device} for training')


    """ Data Type 셋팅 """
    if args.data == 'ILSVRC2012':
        """ The same environment as the setting in the paper """
        batch_size = args.batch_size
        if w_bit == 32 and a_bit == 32:
            EPOCHS = args.epoch
            interval_lr = 0
            weight_lr = 4e-1
            milestones = [30, 60, 85, 95, 105]
        else:
            EPOCHS = args.epoch
            interval_lr = 4e-4
            weight_lr = 4e-2
            milestones = [20, 40, 60, 80]
        train_set = ImageFolder(args.image_dir, transform=custom_transfroms_imagenet(True))
        val_set = ImageFolder(args.image_dir, transform=custom_transfroms_imagenet(False))
        model = resnet18(args.pretrain_model_path, num_classes=1000)

    elif args.data == 'CIFAR10':
        """ 내가 설정한 임의의 환경 """
        batch_size = 128
        if w_bit == 32 and a_bit == 32:
            EPOCHS = args.epoch
            interval_lr = 0
            weight_lr = 1e-1
            milestones = [150, 200,250,275]
        else:
            EPOCHS = args.epoch
            interval_lr = 1e-4
            weight_lr = 1e-2
            milestones = [150, 200,250,275]
        train_set = CIFAR10('/home/zhoukang/quantization_code/Quantization_Code_Pytorch_SSH/dataset/data_cifar10', train=True, transform=custom_transfroms_cifar10(True), download=True)
        val_set = CIFAR10('/home/zhoukang/quantization_code/Quantization_Code_Pytorch_SSH/dataset/data_cifar10', train=False, transform=custom_transfroms_cifar10(False))
        if args.model == "resnet20":
            model = resnet20(args.pretrain_model_path, num_classes=10)
        elif args.model == "resnet32":
            model = resnet32(args.pretrain_model_path, num_classes=10)
        elif args.model == "vgg":
            model = vgg(args.pretrain_model_path, num_classes=10)
    else:
        raise NotImplementedError()

    train_sampler = None

    if w_bit != 32:
        model = set_bit(model, args.w_bit,args.a_bit)

    if args.device is None:
        if w_bit != 32 and a_bit != 32:
            initialize_param(model, train_set, train_sampler, interval_lr, weight_lr)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD([{'params': interval_param_list(model), 'lr': interval_lr},
                           {'params': weight_param_list(model), 'lr': weight_lr}], momentum=0.9, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    """ 멀티프로세싱 분산처리 여부에 따른 모델 설졍 """
    if args.device is not None:
        model = model.to(device)
    else:
        model = DataParallel(model, device_ids=args.deviceIds).to(device=args.device)

    print('model type is DataParallel')

    print('workers', args.workers)
    print('batch_size', args.batch_size)


    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.workers,
                              shuffle=True,pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=100, num_workers=args.workers, shuffle=False,
                            pin_memory=True)

    cudnn.benchmark = True

    # .........................设置需要记录的参数列表.........................#
    epoch1 = []
    val_loss = []
    val_top1 = []
    val_top5 = []
    column_names = ["epoch", "val_loss", "val_top1", "val_top5"]

    current_time = get_current_time()
    if not os.path.exists(args.csv):
        os.makedirs(args.csv)
    csv_file = str(args.csv) + str(args.model) + '_' + str(args.data) + "_" + "QIL "+ '_' + str(args.w_bit) + 'bit' + '_' + str(current_time) + ".csv"

    best_acc = .0
    model_logger.info(f'{args.exp}_Start: {w_bit} | {a_bit}')
    param_logger.info(f'{args.exp}_Start: {w_bit} | {a_bit}')

    for EPOCH in range(1, EPOCHS + 1):
        epoch1.append(EPOCH)
        """ EPOCH마다 GPU 할당되는 인덱스 리스트 설정 """
        train(model, criterion, optimizer, lr_scheduler, train_loader, EPOCH, args)

        # evaluate on test set
        losses_val, prec1, prec5 = validate(val_loader, model, criterion)
        val_loss.append(losses_val)
        val_top1.append(prec1)
        val_top5.append(prec5)

        # remember best precision and save checkpoint
        is_best = prec1 > best_prec
        best_prec = max(prec1, best_prec)
        if EPOCH % 100 == 0:
            write_lists_to_csv(csv_file, [epoch1, val_loss, val_top1, val_top5],column_names)
        print('best acc.................: {:1f}'.format(best_prec))
        if EPOCH%10 == 0:
            print("使用QIL量化方法训练，bit为：", args.w_bit)
        if is_best == True:
            check_point = {'state_dict': model.state_dict(),
                           'optim': optimizer.state_dict(),
                           'EPOCH': EPOCH,
                           'Acc': best_acc}
            torch.save(check_point, args.log + args.model + '_'+args.data  + '_best_{}_{}_model.pt'.format(args.w_bit,args.a_bit))
    write_lists_to_csv(csv_file, [epoch1, val_loss, val_top1, val_top5], column_names)
    model_logger.info(f'Best_Acc: {best_acc:.4f}%')


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Ex")
    parser.add_argument('-d', '--data', type=str, default='CIFAR10', help='ILSVRC2012 or CIFAR10 or CIFAR100')
    parser.add_argument('-b', '--batch-size', type=int, default=256, help='batch size')
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--multiprocessing-distributed', action='store_true', help='병렬처리를 위해 Multi-GPU 사용')
    parser.add_argument('-id', '--device', default='0', type=str, help='gpu device')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:54321', type=str, help='분산처리를 사용하기 위해 사용되는 URL')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--world-size', default=1, type=int, help='분산처리를 하기 위한 노드의 수')
    parser.add_argument('--pretrained-path', type=str, default='./best_32_32_model.pt', help='Pretrained Model의 경로 지정')
    parser.add_argument('--image-dir', '-i', type=str, default='../imagenet/', help='imagenet data dir')
    parser.add_argument('--exp', type=str, default='qil', help='quantization method')
    parser.add_argument('--pretrain_model_path', type=str, default='', help='')
    parser.add_argument('--print_freq', type=int, default=100, help='')
    parser.add_argument('--w_bit', default=5, type=int, help='')
    parser.add_argument('--a_bit', default=5, type=int, help='')
    parser.add_argument('--seed', default=5, type=int, help='')
    parser.add_argument('--epoch', default=300, type=int, help='')
    parser.add_argument('--model', type=str, default='resnet20', help='')
    parser.add_argument('--csv', type=str, default='result/csv/', help='quantization method')
    parser.add_argument('--log', type=str, default='result/log/', help='quantization method')
    args = parser.parse_args()
    args.device = "1"
    device = torch.device('cuda' + ':' + args.device if torch.cuda.is_available() else 'cpu')
    config = [["resnet32", 3],
              ["resnet32", 2],
              ]
    model_logger.info(f'Experiment: {args.exp}')
    for model,bit in config:
        args.model = model
        args.w_bit = bit
        args.a_bit = bit
        nan = main(args)

