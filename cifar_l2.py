import torch
import numpy as np
import os
import os.path as osp
import torchvision
from dataset.cifar import CIFAR10
from torch.utils.data import dataloader
from torchvision.datasets import cifar

""" 
    cifar-10 每一个test数据 与所有train数据的l2距离并且存入一个文件
"""

GPU_ID = 1
ROOT_DIR = osp.abspath(osp.dirname(__file__))
DATA_ROOT_DIR = osp.join(ROOT_DIR, 'data', 'cifar')
SAVE_DIR = osp.join(ROOT_DIR, 'save_dt')
if not osp.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)

device = torch.device('cuda', GPU_ID) if torch.cuda.is_available() else torch.device('cpu')

def transform_test(img):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    img = transform(img)
    return img


if __name__ == "__main__":
    data_set = 'cifar_10'
    save_dir = osp.join(SAVE_DIR, data_set)
    if not osp.exists(save_dir):
        os.mkdir(save_dir)
    cifar_10_train = CIFAR10(DATA_ROOT_DIR, train=True, transform=transform_test, download=False)
    cifar_10_test = CIFAR10(DATA_ROOT_DIR, train=False, transform=transform_test, download=False)
    train_dataloader = dataloader.DataLoader(cifar_10_train, batch_size=500)
    test_dataloader = dataloader.DataLoader(cifar_10_test, batch_size=1)

    for _, (img_test, label_test, index_test) in enumerate(test_dataloader, start=1):
        print(f'get test_dis - index : {index_test[0]}')
        save_distance = torch.zeros(len(cifar_10_train), device=device)
        for _, (img_train, label_train, index_train) in enumerate(train_dataloader, start=1):
            img_test = img_test.to(device)
            img_train = img_train.to(device)
            distance = torch.sum(torch.square((img_test - img_train)), dim=(1, 2, 3))
            for i in range(index_train.size(0)):
                save_distance[index_train[i]] = distance[i]
        torch.save(save_distance.cpu(), osp.join(save_dir, f"test_{index_test[0]}.pkl"))
