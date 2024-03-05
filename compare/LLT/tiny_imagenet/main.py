# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""
import math
import os
import sys
import argparse
import time
from datetime import datetime
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from quant_layer import QuantConv2d

from my_dataset import load_tinyimagenet
from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights,adjust_alpha,set_bit,write_lists_to_csv,get_current_time,time_compute,AverageMeter,accuracy

parser = argparse.ArgumentParser()
parser.add_argument('--net', type=str, default="resnet18", help='')
parser.add_argument('--quant', type=str, default="llq", help='')
parser.add_argument('--device', type=str, default="4", help='')
parser.add_argument('--bit', type=int, default=4, help='')
parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('-resume', action='store_true', default=False, help='resume training')
parser.add_argument('--seed', default=2, type=int, help='Generate random weighted data')
parser.add_argument('--dataset', type=str, default="tiny_imagenet", help='')
parser.add_argument('--workers', type=int, default=4)

def train(epoch,net,args,cifar100_training_loader,optimizer,writer,loss_function,warmup_scheduler):
    train_loss = AverageMeter()
    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
        # update tau
        tau = max(1 * (0.001 ** ((epoch * 392 + batch_index) / 40 / 392)), 0.001)
        for m in net.modules():
            if hasattr(m, '_update_tau'):
                m.tau = tau
        if args.device:
            labels = labels.to(device)
            images = images.to(device)

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        if math.isnan(loss):
            return False
        train_loss.update(loss.item(),images.size(0))
        loss.backward()
        optimizer.step()


        if batch_index % 100 == 0:
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                loss.item(),
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * args.b + len(images),
                total_samples=len(cifar100_training_loader.dataset)
            ))

        if epoch <= args.warm:
            warmup_scheduler.step()

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
    return train_loss.avg

@torch.no_grad()
def eval_training(net,args,cifar100_test_loader,writer,loss_function,epoch=0, tb=True):
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()

    start = time.time()
    net.eval()

    for m in net.modules():
        if hasattr(m, '_quantization'):
            m._quantization()

    for (images, labels) in cifar100_test_loader:

        if args.device:
            images = images.to(device)
            labels = labels.to(device)

        outputs = net(images)
        loss = loss_function(outputs, labels)
        losses.update(loss.item(), images.size(0))

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs, labels, topk=(1, 5))
        top1.update(prec1.item(), images.size(0))
        top5.update(prec5.item(), images.size(0))

    finish = time.time()
    if args.device:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, top1 Accuracy: {:.4f},top5 Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,losses.avg,top1.avg,top5.avg,finish - start))
    print()

    return top1.avg,top5.avg

def main(args):
    begin_time = get_current_time()
    print("开始训练{}模型，量化bit为{}".format(args.net,args.bit))
    global best_prec
    best_prec = 0

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    if args.bit == 32:
        args.quant = "conv"
    else:
        args.quant = "llq"

    net = get_network(args).to(device)

    if args.quant != "conv":
        net = set_bit(net, args.bit)

    dataset_train,dataset_val,num_class = load_tinyimagenet(args)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES,
                                                     gamma=0.2)  # learning rate decay
    iter_per_epoch = len(dataset_train)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)

    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    # use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    # since tensorboard can't overwrite old values
    # so the only way is to create a new tensorboard log
    writer = SummaryWriter(log_dir=os.path.join(
        settings.LOG_DIR, args.net, settings.TIME_NOW))
    input_tensor = torch.Tensor(1, 3, 32, 32)

    # create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    if args.resume:
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_prec = eval_training(tb=False)
            print('best acc is {:0.2f}'.format(best_prec))

        recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
    current_time = get_current_time()
    if args.quant == "conv":
        csv_file = "csv/" + str(args.net) + '_' + str(args.dataset) + "_" + 'float' + '_' + str(current_time) + '_' + str(args.seed)+ '.csv'
    else:
        csv_file = "csv/" + str(args.net) + '_' + str(args.dataset) + "_" + str(args.quant) + '_' + str(args.bit) + 'bit' + '_' + str(current_time) + '_' + str(args.seed) + ".csv"
    # .........................设置需要记录的参数列表.........................#
    epoch1 = []
    train_losses = []
    val_top1 = []
    val_top5 = []
    column_names = ["epoch", "train_loss","val_top1", "val_top5"]

    for epoch in range(1, settings.EPOCH + 1):
        epoch1.append(epoch)
        if epoch > args.warm:
            train_scheduler.step()

        if args.resume:
            if epoch <= resume_epoch:
                continue

        train_loss = train(epoch,net,args,dataset_train,optimizer,writer,loss_function,warmup_scheduler)
        val_prec1,val_prec5 = eval_training(net,args,dataset_val,writer,loss_function,epoch=epoch)
        if math.isnan(train_loss):
            return True
        else:
            pass
        train_losses.append(train_loss)
        val_top1.append(val_prec1)
        val_top5.append(val_prec5)

        best_prec = max(best_prec,val_prec1)
        print("best prec is ........................................:",best_prec)

        # start to save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[-2] and best_prec < val_prec1:
            weights_path = checkpoint_path.format(net=args.net, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_prec = val_prec1
            continue
        if epoch % 10 == 0:
            write_lists_to_csv(csv_file, [epoch1, train_losses, val_top1, val_top5],
                               column_names)
        if epoch % 10 == 1:
            print("正在训练{}模型的{}bit量化实验".format(args.net,args.bit))
            for m in net.modules():
                if isinstance(m, QuantConv2d):
                    m.show_params()
    val_top1.append(best_prec)
    write_lists_to_csv(csv_file, [epoch1, train_losses, val_top1, val_top5],
                       column_names)
    end_time = get_current_time()
    total_time = time_compute(begin_time,end_time)
    print("训练一个任务代码总共耗时：", total_time)

    return False

if __name__ == '__main__':
    begin_time = get_current_time()
    args = parser.parse_args()
    device = torch.device('cuda' + ':' + args.device if torch.cuda.is_available() else 'cpu')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    config = [["resnet18", 8],
              ["resnet34", 8],
              ["resnet34", 4],
              ["resnet34", 3]
              ]  # model,bit
    for model,bit in config:
        args.net = model
        args.bit = bit
        nan = main(args)

    end_time = get_current_time()
    total_time = time_compute(begin_time, end_time)
    print("训练所有任务代码总共耗时：", total_time)