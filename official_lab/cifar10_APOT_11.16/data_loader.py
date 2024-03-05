import torchvision
import torchvision.transforms as transforms

from resnet import *

def cifar10_loader(args):
    print('=> loading cifar10 data...')
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
    train_dataset = torchvision.datasets.CIFAR10(
        root=args.data_root,
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
        root=args.data_root,
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

    return trainloader,testloader