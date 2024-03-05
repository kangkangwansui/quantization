import argparse

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
#..................关于数据集的参数...................#
parser.add_argument('--num_workers', default=4, type=int, help='')
parser.add_argument('--dataset', default="cifar100", type=str, help='the bit-width of the quantized network')
parser.add_argument('--batch_size', default=64, type=int, help='')
parser.add_argument('--data_root', default='/home/zhoukang/quantization_code/dataset/data_Cifar100/', type=str, help='the dir of dataset')
#.......................量化配置参数........................#
parser.add_argument('-m', '--model', metavar='MODEL', default='resnet18',help='models have resnet18、34')
parser.add_argument('--quant', default='clip_non', type=str,help='quantize mode include : clip_melt、non、clip_non')
parser.add_argument('--bit', default=4, type=int, help='the bit-width of the quantized network')
#.......................优化器参数..........................#
parser.add_argument('--warm', default=1, type=int, help='')
parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--milestone', default=[100,180,240], type=list, help='')
parser.add_argument('--container', default='', type=str, help='')
#........................实验结果保存设置..........................#
parser.add_argument('--csv_a', default='result/cifar100_csv_a/', type=str, help='csv_a保存的路径')
parser.add_argument('--csv', default='result/cifar100_csv/', type=str, help='csv保存的路径')
parser.add_argument('--log', default='result/cifar100_log/', type=str, help='log保存的路径')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
#........................实验主函数参数设置..........................#
parser.add_argument('-id', '--device', default='1', type=str, help='gpu device')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--print-freq', '-p', default=100, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--seed', default=7, type=int, help='Generate random weighted data')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--float', default=True, type=bool, help='')

args = parser.parse_args()