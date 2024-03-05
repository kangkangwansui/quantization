import os
import sys
import csv
import time
import shutil
import datetime
import numpy as np

from resnet import *

from quant_layer import CONV

def get_network(args):
    """ return given network
    """
    if args.arch == 'resnet20':
        from resnet import resnet20
        net = resnet20(float=args.float)
        print("使用{}模型训练，量化bit为{}".format(args.arch,args.bit))
    elif args.arch == 'resnet32':
        from resnet import resnet32
        net = resnet32(float=args.float)
        print("使用{}模型训练，量化bit为{}".format(args.arch,args.bit))
    elif args.arch == 'vgg':
        from vgg import vgg
        net = vgg()
        print("使用{}模型训练，量化bit为{}".format(args.arch, args.bit))
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

def set_quant_config(model,args):
    for m in model.modules():
        if isinstance(m, CONV):
            m.quant_mode = args.quant
            m.bit = args.bit
            m.bit_range = 2 ** args.bit
    return model


def get_current_time():
    current_time = datetime.datetime.now()
    return current_time.strftime("%Y-%m-%d-%H:%M:%S")

def save_config(args):
    current_time = get_current_time()
    log_file = args.log + str(args.arch) + '_' + str(args.dataset)
    csv_file = args.csv + str(args.arch) + '_' + str(args.dataset)
    csv_a_file = args.csv_a + str(args.arch) + '_' + str(args.dataset)
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
        data_time.update(time.time() - end)

        input, target = input.to(device), target.to(device)

        output = model(input)
        loss = criterion(output, target)

        prec = accuracy(output, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


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

        if i % 100 == 0:
            if args.quant == 'clip_non' or args.quant == 'non':
                for m in model.modules():
                    if isinstance(m, CONV):
                        m.alpha1.data, m.alpha2.data = adjust_alpha(m.weight, m.weight.grad)
    return losses.avg

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
            prec,prec5 = accuracy(output, target,topk=(1,5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec.item(), input.size(0))
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
        shutil.copyfile(filepath, os.path.join(fdir, 'model_best.pth'))

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

def adjust_learning_rate(optimizer, epoch):
    adjust_list = [100, 180, 240]
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

def write_list_and_matrix_to_csv(list1, matrix, filename):
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

    if epoch >= (args.epochs * 0.98):
        write_list_and_matrix_to_csv(column_names,matrix,dir)

    if epoch == (args.epochs - 1):
        write_list_and_matrix_to_csv(column_names, matrix, dir)

def make_conv_grad_matrix(model,epoch,loss,dir):
    conv_layer_grad = model.layer2[1].conv2.weight.grad.cpu().numpy().flatten()
    conv_layer_grad = np.concatenate(([loss],conv_layer_grad))
    conv_layer_grad = np.concatenate(([epoch], conv_layer_grad))

    with open(dir, 'r', newline='') as file:
        reader = csv.reader(file)
        lines = list(reader)

    # 在指定行插入数据
    lines.insert(epoch, conv_layer_grad)

    # 将更新后的内容写回到CSV文件中
    with open(dir, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(lines)

    if epoch == (args.epochs-1):
        print("梯度已经成功写入：" , dir)


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
        print('正在利用{}模型，在数据集{}上训练{}bit的{}方法'.format(args.arch,args.dataset,args.bit,args.quant))

def load_model(model,dir):
    check = torch.load(dir + '/model_best.pth')
    model.load_state_dict(check['state_dict'])
    print("成功加载模型")
    return model


