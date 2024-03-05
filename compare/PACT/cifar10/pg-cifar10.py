from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import utils

import numpy as np
import os, time, sys
import argparse
import csv

import utils.pg_utils as q

#torch.manual_seed(123123)

#########################
# parameters 

batch_size = 256
num_epoch = 300
_LAST_EPOCH = -1 #last_epoch arg is useful for restart
_WEIGHT_DECAY = 1e-4
this_file_path = "result/log/"
save_folder = os.path.join(this_file_path, 'save_CIFAR10_model')
#########################


#----------------------------
# Argument parser.
#----------------------------
parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--save', '-s', action='store_true', help='save the model')
parser.add_argument('--test', '-t', action='store_true', help='test only')
parser.add_argument('--path', '-p', type=str, default=None, help='saved model path')
parser.add_argument('--which_gpus', '-gpu', type=str, default='0', help='which gpus to use')
parser.add_argument('--arch', '-ar', type=str, default="vgg", help='')

# quantization
parser.add_argument('--wbits', '-w', type=int, default=4, help='bitwidth of weights')
parser.add_argument('--abits', '-a', type=int, default=4, help='bitwidth of activations')
parser.add_argument('--ispact', '-pact', action='store_true', help='activate PACT ReLU')

# PG specific arguments
parser.add_argument('--pbits', '-pb', type=int, default=4, help='bitwidth of predictions')
parser.add_argument('--gtarget', '-gtar', type=float, default=0.0, help='gating target')
parser.add_argument('--sparse_bp', '-spbp', action='store_true', help='sparse backprop of PGConv2d')
parser.add_argument('--ispg', '-pg', action='store_true', help='activate precision gating')
parser.add_argument('--sigma', '-sg', type=float, default=0.001, help='the penalty factor')

args = parser.parse_args()

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
#----------------------------
# Load the CIFAR-10 dataset.
#----------------------------

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
def load_cifar10():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize
        ])
    transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    normalize
        ])

    # pin_memory=True makes transfering data from host to GPU faster
    trainset = torchvision.datasets.CIFAR10(root='/home/zhoukang/quantization_code/dataset/data_cifar10/', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=4, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='/home/zhoukang/quantization_code/dataset/data_cifar10/', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=4, pin_memory=True)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes


#----------------------------
# Define the model.
#----------------------------
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

def generate_model(model_arch):
    if model_arch == 'resnet-20':
        if args.ispg:
            import model.pg_cifar10_resnet as m
            kwargs = {'wbits':args.wbits, 'abits':args.abits, \
                      'pred_bits':args.pbits, 'sparse_bp':args.sparse_bp, \
                      'pact':args.ispact}
            return m.resnet20(**kwargs)
        else:
            import model.quantized_cifar10_resnet as m
            kwargs = {'wbits':args.wbits, 'abits':args.abits, 'pact':args.ispact}
            return m.resnet20(**kwargs)
    elif model_arch == 'resnet-32':
        if args.ispg:
            import model.pg_cifar10_resnet as m
            kwargs = {'wbits':args.wbits, 'abits':args.abits, \
                      'pred_bits':args.pbits, 'sparse_bp':args.sparse_bp, \
                      'pact':args.ispact}
            return m.resnet32(**kwargs)
        else:
            import model.quantized_cifar10_resnet as m
            kwargs = {'wbits':args.wbits, 'abits':args.abits, 'pact':args.ispact}
            return m.resnet32(**kwargs)
    elif model_arch == 'vgg':
        if args.ispg:
            from model.vgg import vgg

            return vgg(args.wbits)
        else:
            from model.vgg import vgg

            return vgg(args.wbits)
    else:
        raise NotImplementedError("Model architecture is not supported.")



#----------------------------
# Train the network.
#----------------------------

def train_model(trainloader, testloader, net, device,csv_file):
    global best_prec
    best_prec = 0
    epoches = []
    t_top1 = []
    t_top5 = []
    column_names = ["epoches", "t_top1", "t_top5"]
    if torch.cuda.device_count() > 1:
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        print("Activate multi GPU support.")
        net = nn.DataParallel(net)
    net.to(device)
    # define the loss function
    criterion = (nn.CrossEntropyLoss().cuda() 
                if torch.cuda.is_available() else nn.CrossEntropyLoss())
    # Scale the lr linearly with the batch size. 
    # Should be 0.1 when batch_size=128
    initial_lr = 0.1 * batch_size / 128
    # initialize the optimizer
    optimizer = optim.SGD(net.parameters(), 
                          lr=initial_lr, 
                          momentum=0.9,
                          weight_decay=_WEIGHT_DECAY)
    # multiply the lr by 0.1 at 100, 150, and 200 epochs
    div = num_epoch // 4
    lr_decay_milestones = [div*2, div*3]
    scheduler = optim.lr_scheduler.MultiStepLR(
                        optimizer, 
                        milestones=lr_decay_milestones, 
                        gamma=0.1,
                        last_epoch=_LAST_EPOCH)

    for epoch in range(num_epoch): # loop over the dataset multiple times
        epoches.append(epoch)
        # set printing functions
        batch_time = utils.AverageMeter('Time/batch', ':.3f')
        losses = utils.AverageMeter('Loss', ':6.2f')
        top1 = utils.AverageMeter('Acc', ':6.2f')
        progress = utils.ProgressMeter(
                        len(trainloader),
                        [losses, top1, batch_time],
                        prefix="Epoch: [{}]".format(epoch+1)
                        )

        # switch the model to the training mode
        net.train()

        print('current learning rate = {}'.format(optimizer.param_groups[0]['lr']))
        
        # each epoch
        end = time.time()
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            for name, param in net.named_parameters():
                if 'threshold' in name:
                    loss += args.sigma * torch.norm(param-args.gtarget)
            loss.backward()
            optimizer.step()

            # measure accuracy and record loss
            _, batch_predicted = torch.max(outputs.data, 1)
            batch_accu = 100.0 * (batch_predicted == labels).sum().item() / labels.size(0)
            losses.update(loss.item(), labels.size(0))
            top1.update(batch_accu, labels.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 50 == 49:    
                # print statistics every 100 mini-batches each epoch
                progress.display(i) # i = batch id in the epoch

        # update the learning rate
        scheduler.step()

        # print test accuracy every few epochs
        print('epoch {}'.format(epoch+1))
        t_prec1,t_prec5 = test_accu(testloader, net, device)
        t_top1.append(t_prec1)
        t_top5.append(t_prec5)
        best_prec = max(best_prec,t_prec1)
        print("best_prec is ................:",best_prec)
        if epoch % 100 == 0:
            write_lists_to_csv(csv_file, [epoches,t_top1,t_top5],column_names)
    write_lists_to_csv(csv_file, [epoches, t_top1, t_top5], column_names)



    # save the model if required
    if args.save:
        print("Saving the trained model.")
        utils.save_models(net.state_dict(), save_folder, suffix=args.arch)

    print('Finished Training')


#----------------------------
# Test accuracy.
#----------------------------

def test_accu(testloader, net, device):
    top1 = AverageMeter()
    top5 = AverageMeter()
    net.to(device)
    cnt_out = np.zeros(9) # this 9 is hardcoded for ResNet-20
    cnt_high = np.zeros(9) # this 9 is hardcoded for ResNet-20
    num_out = []
    num_high = []
    def _report_sparsity(m):
        classname = m.__class__.__name__
        if isinstance(m, q.PGConv2d):
            num_out.append(m.num_out)
            num_high.append(m.num_high)

    correct = 0
    total = 0
    # switch the model to the evaluation mode
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            prec1,prec5 = accuracy(outputs, labels,topk=(1,5))
            top1.update(prec1.item(), images.size(0))
            top5.update(prec5.item(), images.size(0))
        print("val_prec is :",top1.avg)
    return top1.avg,top5.avg



#----------------------------
# Test accuracy per class
#----------------------------

def per_class_test_accu(testloader, classes, net, device):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(10):
        print('Accuracy of %5s : %.1f %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


#----------------------------
# Main function.
#----------------------------

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.which_gpus
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Available GPUs: {}".format(torch.cuda.device_count()))

    print("Create {} model.".format(args.arch))
    net = generate_model(args.arch)
    #print(net)

    if args.path:
        print("@ Load trained model from {}.".format(args.path))
        net.load_state_dict(torch.load(args.path))

    print("Loading the data.")
    trainloader, testloader, classes = load_cifar10()

    csv_file = "result/csv/" + args.arch + "_cifar10_" + str(args.wbits) + "bit" + ".csv"
    if args.test:
        print("Mode: Test only.")
        t_top1,t_top5 = test_accu(testloader, net, device)
    else:
        print("Start training.")
        print("开始训练PACT量化方法，模型为{}，量化bit为{}".format(args.arch,args.wbits))
        train_model(trainloader, testloader, net, device,csv_file)
        per_class_test_accu(testloader, classes, net, device)


if __name__ == "__main__":
    args = parser.parse_args()
    args.which_gpus = '3'
    device = torch.device('cuda' + ':' + args.which_gpus if torch.cuda.is_available() else 'cpu')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.which_gpus
    config = [["resnet-32", 4],
              ["resnet-32", 3],
              ["resnet-32", 2]
              ]  # model,bit
    for model, bit in config:
        args.arch = model
        args.wbits = bit
        args.abits = bit
        args.pbits = bit
        main(args)






#############################
# Backup code.
#############################

'''
#----------------------------
# Show images in the dataset.
#----------------------------

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
'''

