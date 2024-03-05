import logging
import math
from pathlib import Path

import torch as t
import yaml
import os

import process
import quan
import util
from model import create_model
import random
import torch
from data_load import load_tinyimagenet


def main(args,script_dir):
    global best_prec
    best_prec = 0

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    output_dir = script_dir / args.output_dir
    output_dir.mkdir(exist_ok=True)

    log_dir = util.init_logger(args.name, output_dir, script_dir / 'logging.conf')
    logger = logging.getLogger()

    with open(log_dir / "args.yaml", "w") as yaml_file:  # dump experiment config
        yaml.safe_dump(args, yaml_file)

    pymonitor = util.ProgressMonitor(logger)
    tbmonitor = util.TensorBoardMonitor(logger, log_dir)
    monitors = [pymonitor, tbmonitor]

    print("使用模型：{}训练，量化bit为：权重{}bit，激活{}bit".format(args.arch,args.quan.weight.bit,args.quan.act.bit))

    if args.device.type == 'cpu' or not t.cuda.is_available() or args.device.gpu == []:
        args.device.gpu = []
    else:
        available_gpu = t.cuda.device_count()
        for dev_id in args.device.gpu:
            if dev_id >= available_gpu:
                logger.error('GPU device ID {0} requested, but only {1} devices available'
                             .format(dev_id, available_gpu))
                exit(1)
        # Set default device in case the first one on the list
        t.cuda.set_device(args.device.gpu[0])
        # Enable the cudnn built-in auto-tuner to accelerating training, but it
        # will introduce some fluctuations in a narrow range.
        t.backends.cudnn.benchmark = True
        t.backends.cudnn.deterministic = False

    # Initialize data loader
    train_loader, val_loader, num_classes = load_tinyimagenet(args)

    model = create_model(args)

    modules_to_replace = quan.find_modules_to_quantize(model, args.quan)
    model = quan.replace_module_by_names(model, modules_to_replace)
    tbmonitor.writer.add_graph(model, input_to_model=train_loader.dataset[0][0].unsqueeze(0))
    logger.info('Inserted quantizers into the original model')

    if args.device.gpu and not args.dataloader.serialized:
        model = t.nn.DataParallel(model, device_ids=args.device.gpu)

    model.to(args.device.type)

    start_epoch = 0
    if args.resume.path:
        model, start_epoch, _ = util.load_checkpoint(
            model, args.resume.path, args.device.type, lean=args.resume.lean)

    # Define loss function (criterion) and optimizer
    criterion = t.nn.CrossEntropyLoss().to(args.device.type)

    # optimizer = t.optim.Adam(model.parameters(), lr=args.optimizer.learning_rate)
    optimizer = t.optim.SGD(model.parameters(),
                            lr=args.optimizer.learning_rate,
                            momentum=args.optimizer.momentum,
                            weight_decay=args.optimizer.weight_decay)
    lr_scheduler = None
    logger.info(('Optimizer: %s' % optimizer).replace('\n', '\n' + ' ' * 11))
    logger.info('LR scheduler: %s\n' % lr_scheduler)

    perf_scoreboard = process.PerformanceScoreboard(args.log.num_best_scores)

    current_time = process.get_current_time()
    log = "result/log/"
    csv = log = "result/csv/"
    if float == True:
        if not os.path.exists(log):
            os.makedirs(log)
        fdir = str(log) + str(args.arch) + '_' + "cifar10" + "_" + 'float' + '_' + str(current_time)  + '.csv'
        if not os.path.exists(fdir):
            os.makedirs(fdir)

        if not os.path.exists(csv):
            os.makedirs(csv)
        csv_file = str(csv) + str(args.arch) + '_' + "cifar10" + "_" + 'float' + '_' + str(current_time) + '.csv'

    else:
        if not os.path.exists(log):
            os.makedirs(log)
        fdir = str(log) + str(args.arch) + '_' + "cifar10" + "_" + "lsq" + '_' + str(args.quan.weight.bit) + 'bit' + '_' + str(current_time) + '_' + ".csv"
        if not os.path.exists(fdir):
            os.makedirs(fdir)

        if not os.path.exists(csv):
            os.makedirs(csv)
        csv_file = str(csv) + str(args.arch) + '_' + "cifar10" + "_" + "lsq"  + '_' + str(args.quan.weight.bit) + 'bit' + '_' + str(current_time) + ".csv"

    # .........................设置需要记录的参数列表.........................#
    epoch1 = []
    train_loss = []
    train_top1 = []
    train_top5 = []
    val_loss = []
    val_top1 = []
    val_top5 = []
    column_names = ["epoch", "train_loss", "train_top1", "train_top5", "val_loss", "val_top1", "val_top5"]



    if args.eval:
        process.validate(val_loader, model, criterion, -1, monitors, args)
    else:  # training
        if args.resume.path or args.pre_trained:
            logger.info('>>>>>>>> Epoch -1 (pre-trained model evaluation)')
            top1, top5, _ = process.validate(val_loader, model, criterion,
                                             start_epoch - 1, monitors, args)
            perf_scoreboard.update(top1, top5, start_epoch - 1)
        for epoch in range(start_epoch, args.epochs):
            epoch1.append(epoch)
            process.adjust_learning_rate(optimizer,epoch)
            logger.info('>>>>>>>> Epoch %3d' % epoch)
            t_top1, t_top5, t_loss = process.train(train_loader, model, criterion, optimizer,
                                                   lr_scheduler, epoch, monitors, args)
            if math.isnan(t_loss):
                return True
            v_top1, v_top5, v_loss = process.validate(val_loader, model, criterion, epoch, monitors, args)
            train_top1.append(t_top1)
            train_top5.append(t_top5)
            train_loss.append(t_loss)
            val_top1.append(v_top1)
            val_top5.append(v_top5)
            val_loss.append(v_loss)
            if v_top1 - best_prec < -10:
                return True

            tbmonitor.writer.add_scalars('Train_vs_Validation/Loss', {'train': t_loss, 'val': v_loss}, epoch)
            tbmonitor.writer.add_scalars('Train_vs_Validation/Top1', {'train': t_top1, 'val': v_top1}, epoch)
            tbmonitor.writer.add_scalars('Train_vs_Validation/Top5', {'train': t_top5, 'val': v_top5}, epoch)

            if epoch % 10 == 0:
                print("正在训练LSQ量化方法的{}模型的{}bit权重，{}bit激活量化".format(args.arch, args.quan.weight.bit, args.quan.act.bit))

            best_prec = max(v_top1, best_prec)

            perf_scoreboard.update(v_top1, v_top5, epoch)
            is_best = perf_scoreboard.is_best(epoch)
            if epoch%100 == 0:
                util.save_checkpoint(epoch, args.arch, model, {'top1': v_top1, 'top5': v_top5}, is_best, args.name, log_dir)
                process.write_lists_to_csv(csv_file, [epoch1, train_loss, train_top1, train_top5, val_loss, val_top1, val_top5],
                                   column_names)
            print('best acc............: {:1f}'.format(best_prec))

        logger.info('>>>>>>>> Epoch -1 (final model evaluation)')
        process.validate(val_loader, model, criterion, -1, monitors, args)
        process.write_lists_to_csv(csv_file, [epoch1, train_loss, train_top1, train_top5, val_loss, val_top1, val_top5],
                                   column_names)

    tbmonitor.writer.close()  # close the TensorBoard
    logger.info('Program completed successfully ... exiting ...')
    logger.info('If you have any questions or suggestions, please visit: github.com/zhutmost/lsq-net')
    return False


if __name__ == "__main__":
    script_dir = Path.cwd()
    args = util.get_config(default_file=script_dir/'config_change.yaml')
    config = [["resnet18", 4],
              ["resnet34", 4]]  # model,bit,
    for model, bit in config:
        args.arch = model
        args.quan.weight.bit = bit
        args.quan.act.bit = bit
        nan = main(args, script_dir)
