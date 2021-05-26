import os
import os.path as osp

import numpy as np
import torch
import torchvision
from torch.utils.data import dataloader

from nets import TestNet
from analysis_lib import analysisNet, analysisResnet
from dataset import cifar

GPU_ID = 0
MAX_EPOCH = 1250
ROOT_DIR = osp.abspath(osp.dirname(__file__))
DATA_ROOT_DIR = osp.join(ROOT_DIR, 'data', 'cifar')
SAVE_DIR = osp.join(ROOT_DIR, 'CIFAR10_SAVE')
if not osp.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)

device = torch.device('cuda', GPU_ID) if torch.cuda.is_available() else torch.device('cpu')


def transform_test(img):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        # torchvision.transforms.Normalize([0, 0, 0], [1, 1, 1])
    ])
    img = transform(img)
    return img


def accuracy(x, classes):
    arg_max = torch.argmax(x, dim=1).long()
    eq = torch.eq(classes, arg_max)
    return torch.sum(eq).float()


def val_net(net, val_dataloader):
    net.eval()
    with torch.no_grad():
        val_accuracy_sum = 0
        for _, (x, y, _) in enumerate(val_dataloader, 1):
            x, y = x.to(device), y.long().to(device)
            x = net(x)
            val_acc = accuracy(x, y)
            val_accuracy_sum += val_acc
        val_accuracy_sum /= len(val_dataloader.dataset)
    return val_accuracy_sum


if __name__ == "__main__":
    SAVE_DIR_NET = osp.join(SAVE_DIR, 'TestNet-1-98ReLU-Bias-Norm-cifar10')
    if not osp.exists(SAVE_DIR_NET):
        os.mkdir(SAVE_DIR_NET)

    cifar_10_train = cifar.CIFAR10(DATA_ROOT_DIR, train=True, transform=transform_test, download=False)
    cifar_10_test = cifar.CIFAR10(DATA_ROOT_DIR, train=False, transform=transform_test, download=False)

    train_dataloader = dataloader.DataLoader(cifar_10_train, batch_size=1024, shuffle=True, num_workers=2)
    test_dataloader = dataloader.DataLoader(cifar_10_test, batch_size=128)

    # net = analysisResnet.resnet18(num_classes=10,nput_size=(3, 32, 32))
    net = TestNet.Cifar10Net(input_size=(3, 32, 32))
    net.to(device)
    ce = torch.nn.CrossEntropyLoss()
    lr = 0.02
    optim = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.90, weight_decay=0.0005)
    val_log_net = open(os.path.join(SAVE_DIR_NET, "net_acc_log.txt"), 'w')
    for i in range(MAX_EPOCH):
        net.train()
        if (i != 0) and (i % 250 == 0):
            lr /= 2
            for param_group in optim.param_groups:
                param_group['lr'] = lr
        for _, (x, y, index) in enumerate(train_dataloader, 1):
            x, y = x.to(device), y.long().to(device)
            x = net(x)
            loss = ce(x, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            acc = accuracy(x, y) / x.size(0)

            # TODO: 保存数据, 保存acc
            print(f"Epoch: {i+1} / {MAX_EPOCH}, Loss: {loss:.4f}, Acc: {acc:.4f}")

        val_acc = val_net(net, test_dataloader)
        print(f"Epoch: {i+1} / {MAX_EPOCH}, Val_Acc: {val_acc:.4f}")
        val_log_net.write(f"Epoch: {i+1} / {MAX_EPOCH}, Val_Acc: {val_acc:.4f}\n")
        val_log_net.flush()
        if (i + 1) % 5 == 0:
            print("Save net....")
            torch.save(net.state_dict(), osp.join(SAVE_DIR_NET, f'net_{i+1}.pth'))
