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
from models.quant_layer import QuantizedConv2d

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights,adjust_alpha,set_bit,write_lists_to_csv,get_current_time,time_compute,AverageMeter,accuracy

parser = argparse.ArgumentParser()
parser.add_argument('--net', type=str, default="resnet20", help='')
parser.add_argument('--quant', type=str, default="pact", help='')
parser.add_argument('--device', type=str, default="1", help='')
parser.add_argument('--bit', type=int, default=4, help='')
parser.add_argument('-b', type=int, default=256, help='batch size for dataloader')
parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('-resume', action='store_true', default=False, help='resume training')
parser.add_argument('--seed', default=7, type=int, help='Generate random weighted data')
parser.add_argument('--dataset', type=str, default="cifar100", help='')

def train(epoch,net,args,cifar100_training_loader,optimizer,writer,loss_function,warmup_scheduler):
    data_time = AverageMeter()
    train_loss = AverageMeter()
    start = time.time()
    net.train()

    end = time.time()
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
        data_time.update(time.time() - end)

        if args.device:
            labels = labels.to(device)
            images = images.to(device)

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        train_loss.update(loss.item(),images.size(0))
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        if batch_index % 100 == 0:
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                loss.item(),
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * args.b + len(images),
                total_samples=len(cifar100_training_loader.dataset)
            ))

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch <= args.warm:
            warmup_scheduler.step()

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

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

    #add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', losses.avg, epoch)
        writer.add_scalar('Test/Accuracy', top1.avg, epoch)

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
        args.quant = "pact"

    net = get_network(args)

    if args.quant != "conv" and args.bit != 32:
        net = set_bit(net, args.bit)

    net = net.to(device)

    # data preprocessing:
    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES,
                                                     gamma=0.2)  # learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
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

        train_loss = train(epoch,net,args,cifar100_training_loader,optimizer,writer,loss_function,warmup_scheduler)
        val_prec1,val_prec5 = eval_training(net,args,cifar100_test_loader,writer,loss_function,epoch=epoch)
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

        if not epoch % settings.SAVE_EPOCH:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)

        if epoch % 100 == 0:
            write_lists_to_csv(csv_file, [epoch1, train_losses, val_top1, val_top5],
                               column_names)
        if epoch % 10 == 1:
            print("正在训练{}模型的{}bit量化实验".format(args.net,args.bit))
            for m in net.modules():
                if isinstance(m, QuantizedConv2d):
                    m.show_params()
    val_top1.append(best_prec)
    write_lists_to_csv(csv_file, [epoch1, train_losses, val_top1, val_top5],
                       column_names)
    writer.close()
    end_time = get_current_time()
    total_time = time_compute(begin_time,end_time)
    print("训练一个任务代码总共耗时：", total_time)

    return False

if __name__ == '__main__':
    begin_time = get_current_time()
    args = parser.parse_args()
    args.device = '1'
    device = torch.device('cuda' + ':' + args.device if torch.cuda.is_available() else 'cpu')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    config = [["resnet18", 8],
              ["resnet18", 5],
              ["resnet18", 4],
              ["resnet34", 8],
              ["resnet34", 5],
              ["resnet34", 4]
              ]  # model,bit
    for model,bit in config:
        args.net = model
        args.bit = bit
        nan = main(args)
    end_time = get_current_time()
    total_time = time_compute(begin_time, end_time)
    print("训练所有任务代码总共耗时：", total_time)