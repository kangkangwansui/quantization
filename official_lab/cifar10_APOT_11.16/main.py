import random

import torch.backends.cudnn as cudnn

from data_loader import cifar10_loader

from utils import *

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

    device = torch.device('cuda' + ':' + args.device if torch.cuda.is_available() else 'cpu')
    float = True if args.bit == 32 else False
    args.float = float
    model = get_network(args)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = get_optimizer(model,args)

    if args.float != True:
        model = set_quant_config(model,args)
    cudnn.benchmark = True

    log_dir,csv_dir,csv_a_dir = save_config(args)

    trainloader,testloader = cifar10_loader(args)


    if args.evaluate:
        model = load_model(model,log_dir)
        val_prec1,val_prec5,val_loss = validate(testloader, model, criterion)
        print("top1 acc is {},top5 acc is {},val_loss acc is {}".format(val_prec1,val_prec5,val_loss))
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        train_loss = train(trainloader, model, criterion, optimizer, epoch)

        make_conv_grad_matrix(model, epoch, train_loss,args.csv + "/grad.csv")

        val_prec1,val_prec5,val_loss = validate(testloader, model, criterion)

        is_best = val_prec1 > best_prec
        best_prec = max(val_prec1, best_prec)
        best_prec5 = max(val_prec5, best_prec5)
        print('best acc: {:1f}'.format(best_prec))

        make_acc_csv(train_loss,val_loss,val_prec1,val_prec5,epoch,best_prec,best_prec5,csv_dir)

        make_alpha_matrix(model,epoch,csv_a_dir)

        show_config(args,epoch)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec': best_prec,
            'optimizer': optimizer.state_dict(),
        }, is_best, log_dir)


if __name__=='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    config = [["resnet20", 32]]
    for model, bit in config:
        args.arch = model
        args.bit = bit
        main(args)