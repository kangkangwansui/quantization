import random

import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from utils import *
from data_loader import get_training_dataloader,get_test_dataloader

def main(args):

    global best_prec,best_prec5
    best_prec = 0
    best_prec5 = 0
    print("使用GPU{}训练".format(args.device))
    print('=> Building model...')

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        torch.backends.cudnn.deterministic = True
        print("随机种子设置为{}".format(args.seed))

    if args.bit == 32:
        args.float = True
        args.container = args.quant
        args.quant = "conv"

    else:
        args.float = False
        if args.quant == "conv":
            args.quant = args.container

    model = get_network(args)

    '''if args.bit != 32:
        model = set_bit(model, args)'''

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = get_optimizer(model,args)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestone,gamma=0.1)  # learning rate decay
    cudnn.benchmark = True


    log_dir,csv_dir,csv_a_dir = save_config(args)

    print('=> loading cifar100 data...')
    trainloader = get_training_dataloader(args)
    testloader = get_test_dataloader(args)
    iter_per_epoch = len(trainloader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)


    if args.evaluate:
        model = load_model(model,args,optimizer)
        val_prec1,val_prec5,val_loss = eval_training(model, args, testloader, criterion,1)
        print("top1 acc is {},top5 acc is {} , val_loss is {}".format(val_prec1,val_prec5,val_loss))
        return

    for epoch in range(args.start_epoch, args.epochs):
        if epoch > args.warm:
            train_scheduler.step()

        losses_train = train(epoch, model, args, trainloader, optimizer,criterion,warmup_scheduler)

        # evaluate on test set
        val_prec1,val_prec5,val_loss = eval_training(model, args, testloader, criterion,epoch)

        is_best = val_prec1 > best_prec
        best_prec = max(val_prec1, best_prec)
        best_prec5 = max(val_prec5, best_prec5)
        print('best acc: {:1f}'.format(best_prec))

        make_acc_csv(losses_train,val_loss,val_prec1,val_prec5,epoch,best_prec,best_prec5,csv_dir)

        make_alpha_matrix(model,epoch,csv_a_dir)

        show_config(args,epoch)

        # remember best precision and save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec': best_prec,
            'optimizer': optimizer.state_dict(),
        }, is_best, log_dir)


if __name__=='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device('cuda' + ':' + args.device if torch.cuda.is_available() else 'cpu')
    config = [
              ["resnet34", 8],
              ["resnet34", 5],
              ["resnet34", 4],
              ["resnet18", 8],
              ["resnet18", 5],
              ["resnet18", 4],
             ]
    for model, bit in config:
        args.model = model
        args.bit = bit
        main(args)